########################################################################################################################
# Author: Lorenzo Ferretti
# Year: 2018
#
# Contains classes Synthesiser and FakeSynthesiser. These are used to invoke Vivado HLS and generate the synthesis data
# given an input configuration. Fake synthesiser retrieve data from the exhaustive exploration already performed
########################################################################################################################
from os import listdir
from os.path import isfile, join, exists
import subprocess
import xml.etree.ElementTree

class FakeSynthesis:
    def __init__(self, entire_ds, lattice):
        self.entire_ds = entire_ds
        self.lattice = lattice

    def synthesise_configuration(self, config):
        c = self.lattice.revert_discretized_config(config)
        result = None
        for i in range(0, len(self.entire_ds)):
            if self.entire_ds[i].configuration == c:
                result = (self.entire_ds[i].latency, self.entire_ds[i].area,i,(self.entire_ds[i].latency != 10000000 and self.entire_ds[i].area!=10000000))
                break
        return result

class VivdoHLS_Synthesis:

    def __init__(self, lattice, ds_description, ds_description_ordered, ds_bundling, project_description):
        self.lattice = lattice
        self.ds_descriptor = ds_description
        self.ds_descriptor_ordered = ds_description_ordered
        self.bundling_sets = ds_bundling
        self.project_name = project_description["prj_name"]
        self.test_bench = project_description["test_bench_file"]
        self.source_folder = project_description["source_folder"]
        self.top_function = project_description["top_function"]

    def synthesise_configuration(self, config):
        c = self.lattice.revert_discretized_config(config)
        script_name = self.generate_tcl_script(c)
        if script_name is None:
            return None, None
        # process = subprocess.Popen(["vivado_hls", "-f", "./exploration_scripts/" + script_name + ".txt", ">>", script_name + ".out"])
        if exists("./"+self.project_name+"/"+script_name+"/syn/report/"+self.top_function+"_csynth.xml"):
            print("File already synthesised and already in folder!")
            pass
        else:
            process = subprocess.Popen("vivado_hls -f ./exploration_scripts/" + script_name + ".txt >> ./exploration_scripts/" + script_name + ".out", shell=True)
            process.wait()
        latency, area = self.get_synthesis_results(script_name)
        return latency, area

    def generate_tcl_script(self, configuration):
        clock = self.ds_descriptor["clock"]
        new_line = " \n"
        script = "open_project " + self.project_name + new_line
        script += "set_top " + self.top_function + new_line
        file_list = []
        test_bench = None
        for f in listdir(self.source_folder):
            if isfile(join(self.source_folder, f)):
                if f != self.test_bench:
                    if f[-2:] == ".c" or f[-2:] == ".h":
                        file_list.append(f)
                else:
                    test_bench = f

        script += "add_files -tb " + self.source_folder + '/' + test_bench + new_line
        for f in file_list:
            script += "add_files " + self.source_folder + '/' + f + new_line

        script += "open_solution sol_" + "_".join(str(e) for e in configuration) + new_line
        script += "set_part {xc7k160tfbg484-1}" + new_line
        if isinstance(clock, dict):
            clock_list = clock["clock"]
            clock_idx = [i for i, tupl in enumerate(self.ds_descriptor_ordered) if tupl[0] == "clock-clock"]
            # clock_idx = self.ds_descriptor.benchmark_directives.index("clock")
            script += "create_clock -period " + str(configuration[clock_idx[0]]) + " -name default" + new_line
        else:
            script += "create_clock -period 10 -name default" + new_line

        # script += "set_directive_interface -mode s_axilite \"" + self.top_function + "\"" + new_line

         # Start exploring all the other configuration except the clock
        for c in xrange(len(configuration)):
            if c == clock_idx[0]:
                continue
            else:
                conf = configuration[c]
                directive = self.ds_descriptor_ordered[c]
                directive = self.add_directive(conf, directive, script)
                if directive is None:
                    return None
                script += directive

        script += "csynth_design" + new_line
        script += "exit" + new_line

        script_file = "sol_"+ "_".join(str(x) for x in configuration)
        outfile = open("./exploration_scripts/"+script_file+".txt", "w")
        outfile.write(script)
        outfile.close()

        return script_file

    def add_directive(self, directive_value, directive, script):
        kind = directive[0].split('-')[0]
        if kind == "unrolling":
            script = self.add_unrolling_directive(directive_value, directive)
        if kind == "bundling":
            script = self.add_bundling_directive(directive_value)
        if kind == "pipelining":
            script = self.add_pipeline_directive(directive_value, directive, script)
            if script is None:
                return None
        if kind == "inlining":
            script = self.add_inlining_directive(directive_value, directive)
        if kind == "partitioning":
            script = self.add_partitioning_directive(directive_value, directive)

        return script

    def add_unrolling_directive(self, directive_value, directive):
        new_line = "\n"
        loop_name = directive[0].split('-')[1]
        script = ""
        if directive_value != 0:
            script += "set_directive_unroll -factor " + str(directive_value) + " \"" + self.top_function + "/" + loop_name \
                      + "\"" + new_line
        else:
            script += "set_directive_loop_flatten -off \"" + self.top_function + "/" + loop_name + "\"" + new_line

        return script

    def add_bundling_directive(self, directive_value):
        new_line = "\n"
        script = ""
        script += "set_directive_interface -mode s_axilite \"" + self.top_function + "\"" + new_line
        bundle_ports = self.bundling_sets[0]
        bundle_sets = self.bundling_sets[1][directive_value]
        for i in range(0, len(bundle_ports)):
            script += "set_directive_interface -mode m_axi -offset direct -bundle " + str(bundle_sets[i]) + " \"" +\
                      self.top_function + "\" " + bundle_ports[i] + new_line
        return script

    def add_pipeline_directive(self, directive_value, directive, old_script):
        new_line = "\n"
        script = ""
        target = directive[0].split('-')[1]
        idx = old_script.find(target)
        if idx >= 0:
            substring = old_script[idx-30:idx]
            unroll = [int(s) for s in substring.split() if s.isdigit()]
            if len(unroll) != 0:
                unroll = unroll.pop()
                if unroll == 9 or unroll == 160 or unroll == 152:
                    return None

        if target.find("loop") >= 0:
            if directive_value != 0:
                script += 'set_directive_pipeline \"' + self.top_function + "/" + target + '\"\n'
            else:
                script += "set_directive_pipeline -off " + "\"" + self.top_function + "/" + target + "\"" + new_line
        else:
            if directive_value != 0:
                script += 'set_directive_pipeline \"' + target + '\"\n'
            else:
                script += "set_directive_pipeline -off " + "\"" + target + "\"" + new_line

        return script

    def add_inlining_directive(self, directive_value, directive):
        new_line = "\n"
        script = ""
        target = directive[0].split('-')[1]
        if directive_value == 0:
            script = script + "set_directive_inline -off \"" + target + "\"" + new_line
        else:
            script = script + "set_directive_inline \"" + target + "\"" + new_line

        return script

    def add_partitioning_directive(self, directive_value, directive):
        new_line = "\n"
        script = ""
        target = directive[0].split('-')[1]
        type = "block"
        if directive_value != 0:
            script += "set_directive_array_partition -type " + type + " -factor " + str(directive_value) + " -dim 1 \""\
                      + self.top_function + "\" " + target + new_line

        return script

    def get_synthesis_results(self, script_name):
        outputfile = open("./exploration_scripts/"+script_name+".out", "r")
        content = outputfile.read()
        if content.find("has been removed") > 0:
            x = 100000
            LUT = 100000
            FF = 1000000
        else:
            x = None
            LUT = None
            FF = None
            root = xml.etree.ElementTree.parse("./"+self.project_name+"/"+script_name+"/syn/report/"+self.top_function+"_csynth.xml").getroot()
            # For each solution extract Area estimation
            for area_est in root.findall('AreaEstimates'):
                for resource in area_est.findall('Resources'):
                    for child in resource:
                        if "FF" in child.tag:
                            FF = int(child.text)

                        if "LUT" in child.tag:
                            LUT = int(child.text)
            # For each solution extract Performance estimation
            for perf_est in root.findall('PerformanceEstimates'):
                for latency in perf_est.findall('SummaryOfOverallLatency'):
                    for child in latency:
                        if "Average-caseLatency" in child.tag:
                            x = int(child.text)
        latency = x
        area = LUT
        return latency, area
