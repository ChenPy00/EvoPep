# -*- coding: utf-8 -*-
from generate_feature import BasicDes
import pandas as pd
from tqdm import tqdm


def cal_pep(peptides, sequence, output_path):
    peptides_descriptors = []
    for peptide in tqdm(peptides):
        if 5 > len(peptide) or len(peptide) > 30:
            continue
        peptides_descriptor = {}
        peptide = str(peptide)
        Basic = BasicDes.cal_discriptors(peptide)
        peptides_descriptor.update(Basic)
        peptides_descriptors.append(peptides_descriptor)

    df = pd.DataFrame(peptides_descriptors,index=sequence.index)
    output_csv = pd.concat([sequence, df], axis=1)
    output_csv.to_csv(output_path)
    # return output_csv




