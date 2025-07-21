import os 
import torch
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

#  Set these things
retain_dim=312

with torch.no_grad():
    eigvec = torch.load(f"pca/eigenvec_mut.pt", map_location='cuda', weights_only=True).float()[:, :retain_dim]
    eigval = torch.load(f"pca/eigenval_mut.pt", map_location='cuda', weights_only=True).float()[:retain_dim]
    mean = torch.load(f"pca/avg1536_mut_0.pt", map_location='cuda', weights_only=True).float()
    sd = torch.load(f"pca/sd1536_mut_0.pt", map_location='cuda', weights_only=True).float()
    
def normalize(tensor):
    return (tensor - mean) / sd

def calc_diff_std(mutants):
    avg = torch.stack(list(mutants.values())).mean(dim=0)
    # print("avg of mutants", avg[:3, :3])
    tensor = torch.zeros(mutants[41].size(), device='cuda', dtype=torch.float32)
    for k in mutants:
        tensor += (mutants[k]-avg)**2
    return torch.sqrt(tensor/len(mutants))

def rediceDim(tensor):
    return tensor @ eigvec

def euclidean(tensor):
    return torch.sqrt((tensor**2 / eigval).mean(dim=-1))

def pad_expand(tensor):
    return torch.concat((tensor, torch.zeros((1650-tensor.size(0)), device='cuda', dtype=torch.float32)), dim=0)
    
def calcDiff(DF, i, seed_start, seed_end, create_dataset=False):
    num_seed = seed_end-seed_start
    if os.path.exists(f"esm3_embeddings/wt/AF-{DF.loc[i, 'Entry']}-F1-model_v4_Emb.pt"):
        wt = torch.load(f"esm3_embeddings/wt/AF-{DF.loc[i, 'Entry']}-F1-model_v4_Emb.pt", map_location='cuda', weights_only=True).float()
        # print("wt", wt[:3, :3])
        
        mutants = {}
        for seed in range(seed_start, seed_end):
            mutants[seed] = torch.load(f"esm3_embeddings/rs{seed}/AF-{DF.loc[i, 'Entry']}-F1-model_v4_Emb_rs{seed}tp06.pt", map_location='cuda', weights_only=True).float()
            # if seed == 41:
            #     print("rs41", mutants[seed][:3, :3])
            # print(mutantsrs[seed].size())
            mutants[seed] -= wt
            # if seed == 41:
            #     print("rs41 diff", mutants[seed][:3, :3])
            # print(mutants[seed].size())
            mutants[seed] = rediceDim(normalize(mutants[seed]))
            # if seed == 41:
            #     print("rs41 PCA", mutants[seed][:3, :3])
            # print(mutants[seed].size())
        del wt
    
        # Calc epistasis
        epi = calc_diff_std(mutants)
        # print("epi", epi[:3, :3])
        
        # Calc mut score by average
        for seed in range(seed_start, seed_end):
            mutants[seed] = euclidean(mutants[seed])
            # print(mutants[seed].size())
        
        mut = torch.stack(list(mutants.values()))
        mut = mut.mean(dim=0)
        # print("mut", mut[:3])
        
        mut = 2*mut / 754.8684692382812 # 1547.5189208984375
        # mut = pad_expand(mut)
        # epi = pad_expand(epi)
        # print("mut", mut.max().item())
        # print("epi", epi.max().item())
        torch.save(mut, f"output_tensors/{DF.loc[i, 'Entry']}_sd{num_seed}_mut.pt")
        # if create_dataset:
        return i, mut, epi
    # else:
    #     return entry
    else:
        # if create_dataset:
        #     return None, None, None, torch.zeros((1650), device='cuda'), torch.zeros((1650), device='cuda')
        return i, None, None

def combine_epi(DF, i, epi, mut, num_seeds):
    seq_len = epi.size(0)
    # print(epi.size())
    # quit()
    epi = euclidean(epi)
    # print("epi", epi[:3])

    epi = 1.5*epi / 336.1826171875 # 445.9331359863281
    epi_min = epi.min().item()
    
    R2 = r2(mut, epi, seq_len)
    slope = normal_equ(mut, epi)
    torch.save(epi, f"output_tensors/{DF.loc[i, 'Entry']}_sd{num_seeds}_epi.pt")
    return epi, R2, epi_min, slope

def r2(x, y, seq_len):
    ssr = ((x-y)**2).sum().item()
    sst = ((y-y.mean())**2).sum().item()
    return 1-ssr*(seq_len-1)/(sst*(seq_len-2))

def normal_equ(x, y):
    x = torch.cat((torch.ones((x.size(0), 1), device='cuda', dtype=torch.float32), x.unsqueeze(-1)), dim=1)
    return torch.linalg.lstsq(x, y).solution[1].item() #torch.linalg.solve(x.T @ x, x.T @ y)[1].item()

def parallel(DF, seed_start, seed_end, create_dataset=True, num_workers=1, file_name='Dataset'):
    num_seed = seed_end - seed_start
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # tensors = list(executor.map(load_tensor, file_list))
        futures = [executor.submit(calcDiff, DF, i, seed_start, seed_end, create_dataset) for i in DF.index] # entry, 
        if create_dataset:
            mut_tensor = []
            epi_tensor = []
        all_entry = []
        all_r2 = []
        epi_min = []
        all_slope = []
        for fut in futures:
            DF_score = pd.DataFrame()
            if create_dataset:
                i, avg, std = fut.result()
                if avg == None:
                    continue
                # mut_tensor.append(avg)
                # epi_tensor.append(std)
                # np.save(f"output_tensors/{entry}_E.npy", epi.detach().cpu().numpy())
                torch.save(std, f"output_tensors/{DF.loc[i, 'Entry']}_sd{num_seed}_epi_separate.pt")
            else:
                i = fut.result()
            std, R2, std_min, slope = combine_epi(DF, i, std, avg, seed_end-seed_start)
            DF_score["M_score"] = pd.Series(avg.detach().cpu().numpy())
            DF_score["E_score"] = pd.Series(std.detach().cpu().numpy())
            DF_score["Sequence"] = pd.Series(list(DF.loc[i, "Sequence_final"]))
            DF_score.to_csv(f"output_dataset/{DF.loc[i, 'Entry']}_sd{num_seed}_MEscores.csv", index=False)

            all_entry.append(DF.loc[i, 'Entry'])
            all_r2.append(R2)
            all_slope.append(slope)
            epi_min.append(std_min)
        # mut_tensor = torch.stack(mut_tensor) 
        # epi_tensor = torch.stack(epi_tensor)
        # # print(mut_tensor.size())
        # torch.save(mut_tensor, f"output_dataset/{file_name}_sd{num_seed}_mut.pt")
        # torch.save(epi_tensor, f"output_dataset/{file_name}_sd{num_seed}_epi.pt")
        # # print(DF)
        # if create_dataset:
        #     DFo = pd.DataFrame()
        #     for i in DF.index:
        #         length = DF.loc[i, 'Length']
        #         DFo[f"{DF.loc[i, 'Entry']}_mut"] = pd.Series(mut_tensor[i, :].tolist())
        #         DFo[f"{DF.loc[i, 'Entry']}_epi"] = pd.Series(epi_tensor[i, :].tolist())
        #     DFo.to_csv(f"output_dataset/{file_name}_sd{num_seed}_raw.csv", index=False)

        DFa = pd.DataFrame()
        DFa["Entry"] = pd.Series(all_entry)
        DFa["R2"] = pd.Series(all_r2)
        DFa["slope"] = pd.Series(all_slope)
        DFa["epi_min"] = pd.Series(epi_min)
        DFa.to_csv(f"output_dataset/{file_name}_sd{num_seed}_analysis.csv", index=False)
        

parser = argparse.ArgumentParser(description="A script that takes an input value.")
parser.add_argument("-file_name", type=str, required=True, help="Name of UniProt-format dataframe")
parser.add_argument("-num_seeds", type=int, required=True, help="Name of random seeds-unique replicates")
parser.add_argument("-num_cpu", type=int, required=True)
args = parser.parse_args()
file_name = args.file_name
num_seeds = args.num_seeds
num_cpu = args.num_cpu

if __name__ == '__main__':
    with torch.no_grad():
        DF = pd.read_csv(f"input_csv/{file_name.replace('.csv', '')}.csv", header=0, \
                        usecols=["Entry", "Sequence_final"])#.loc[0]
        seed_start = 41
        seed_end = seed_start + num_seeds
        parallel(DF, seed_start, seed_end, num_workers=num_cpu, file_name=file_name)
