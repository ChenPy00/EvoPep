# -*- coding: utf-8 -*-
# from . import BasicDes
import BasicDes, Autocorrelation, CTD, PseudoAAC, AAComposition, QuasiSequenceOrder
import pandas as pd
import numpy as np
import sys
import multiprocessing
"""
直接给list形式的氨基酸序列生成
"""
def cal_pep(peptide):

    peptide = str(peptide)
    peptides_descriptor={}
    AAC = AAComposition.CalculateAAComposition(peptide)
    DIP = AAComposition.CalculateDipeptideComposition(peptide)
    MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide, lamba=5)
    CCTD = CTD.CalculateCTD(peptide)
    QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)
    PAAC = PseudoAAC._GetPseudoAAC(peptide,lamda=5)
    APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)
    Basic = BasicDes.cal_discriptors(peptide)
    peptides_descriptor.update(AAC)
    peptides_descriptor.update(DIP)
    peptides_descriptor.update(MBA)
    peptides_descriptor.update(CCTD)
    peptides_descriptor.update(QSO)
    peptides_descriptor.update(PAAC)
    peptides_descriptor.update(APAAC)
    peptides_descriptor.update(Basic)

    return peptides_descriptor