

class GPTSTask(Task):

    def preprocess(self):
        dict_test=(build_dict_test('/public/home/liuxj/liuxj_share/project/GPTS/0.2.1/general/Profile_Rho.txt','/public/home/liuxj/liuxj_share/project/GPTS/0.2.1/general/g900003.00230_ITER_15MA_eqdsk16HR.txt')); #load an example profile
        print(dict_test.keys()) #show profile parameters, 
        #'rho' is the rho coordinate, 
        #'Te', 'TD', 'TT', 'Tath', 'Ne', 'ND', 'NT', 'Nath' are background temperature (Joule) and number density (m^-3) parameters for electron, deuterium, thermal He particles and tritium, respectively,
        # 'Na_source' is the production rate for fusion alpha particles (m^-3s^-1)
        dict_test['gpts_run_directory']='/public/home/xiaojianyuan/data/ITER_test';
        dict_test['num_time_steps']=40000001; #set number of time steps
        dict_test["NUM_PARTICLES_PER_PROC"]=25; #set number of particles per MPI process
        dict_test['num_node']=12; #set number of nodes to run, total number of particles is num_node*3200

    def postprocess(self):
        aph.keys() 

    def execute(self):
        alpha_heat_api_gpts(dict_test) #run GTPS and obtain result alpha profiles


    def run(self):
        self.preprocess()
        self.execute()
        self.postprocess()
