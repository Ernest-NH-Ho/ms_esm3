import re
import pandas as pd
import argparse

## Change the TRANSIT to SIGNAL if needed !!!!!!!!!!!
def extractCS(describe, tag):
    if str(describe).count(tag) > 0:
        start = describe.find(tag) + len(tag)
        describe = describe[start:]
        end = describe.find(';')
        if end!=-1:
            nums = describe[:end].split('..')
        else:
            nums = describe.split('..')
        if '?' in nums[1] or nums[1] == '' or '>' in nums[1] or '<' in nums[1]:
            return 0
        else:
            return int(nums[1])
    else:
        return 0

def expand_ranges(lst):
    expanded_list = []
    for item in lst:
        if re.match(r'^\d+\.\.\d+$', item):  # Match patterns like '10..15'
            start, end = map(int, item.split('..'))
            expanded_list.extend(map(str, range(start, end + 1)))
        else:
            expanded_list.append(item)
    return expanded_list

def extract(txt, cs, tag):
    # Regex pattern to match numbers between "BINDING" and ";"
    ## We have cut all the STP already
    if txt.count(";") > 0:
        pattern1 = fr'{tag} (\d+);'
        pattern2 = fr'{tag} (\d+\.\.\d+);'
    else:
        pattern1 = fr'{tag} (\d+)'  # only for single site
        pattern2 = fr'{tag} (\d+\.\.\d+)'  # only for single site

    # Extract all matching numbers
    binding_numbers1 = list(set(re.findall(pattern1, txt)))
    binding_numbers2 = list(set(re.findall(pattern2, txt)))
    binding_numbers2 = expand_ranges(binding_numbers2) + binding_numbers1
    binding_numbers2 = [str(int(i)-cs) for i in binding_numbers2]
    return binding_numbers2
    
def extract_bind_act_motif(row):
    out = []
    if isinstance(row["Binding site"], str) and row["Binding site"]!="":
        out += extract(txt=row["Binding site"], cs=row["CS"], tag="BINDING")
    if isinstance(row["Active site"], str) and row["Active site"]!="":
        out += extract(txt=row["Active site"], cs=row["CS"], tag="ACT_SITE")
    if isinstance(row["Motif"], str) and row["Motif"]!="":
        out += extract(txt=row["Motif"], cs=row["CS"], tag="MOTIF")
    if isinstance(row["Site"], str) and row["Site"]!="":
        out += extract(txt=row["Site"], cs=row["CS"], tag="SITE")
    return ",".join(out)

def adjustIdx(idx:str, cs:str):
    if idx!='':
        cs = int(cs)
        idx = idx.split(",")
        idx = [str(int(i)+cs) for i in idx]
        return ",".join(idx)
    else:
        return idx

parser = argparse.ArgumentParser(description="A script that takes an input value.")
parser.add_argument("-file_name", type=str, required=True, help="Name of UniProt-format dataframe")
parser.add_argument("-file_type", type=str, required=True, help="csv or tsv")
args = parser.parse_args()
file_name = args.file_name
file_type = args.file_type
if file_type == 'csv':
    sep = ','
elif file_type == 'tsv':
    sep='\t'
    
if __name__ == '__main__':
    DF = pd.read_csv(f"input_csv/{file_name}.{file_type}", header=0, sep=sep).astype(str)
    
    ## Find all CS (cleavage sites of signal/transit peptide)
    DF['CS'] = DF['Signal peptide'].apply(lambda x:extractCS(x, 'SIGNAL ') if x!='' else 0)
    # print(type(DF.loc[0, 'Transit peptide']))
    DF['CS'] = DF[['CS', 'Transit peptide']].apply(lambda x:extractCS(x['Transit peptide'], 'TRANSIT ') \
                                                   if x['CS']==0 or x['Transit peptide']!='nan' else x['CS'], axis=1)
    DF['Sequence'] = DF[['Sequence', 'CS']].apply(lambda x:x['Sequence'][x['CS']:], axis=1)
    DF['Length'] = DF['Sequence'].str.len()
    
    ## Compile all active/binding/motif sites
    DF["Act_bind_motif_sites"] = DF[["Binding site", "Active site", "Motif", "Site", "CS"]].apply(lambda x:extract_bind_act_motif(x), axis=1)

    ## Correct site numbers according to CS (cleavage site)
    # print(DF['Act_bind_motif_sites'])
    DF["Act_bind_motif_sites_wifSTP"] = DF[["Act_bind_motif_sites", "CS"]].apply(lambda x:adjustIdx(x["Act_bind_motif_sites"], x["CS"]), axis=1)
    
    ## Output is always csv
    DF.to_csv(f"input_csv/{file_name}.csv", index=False)
    

    
