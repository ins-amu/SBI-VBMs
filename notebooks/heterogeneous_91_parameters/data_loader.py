import os
from glob import glob
from scipy.io.matlab import loadmat
import numpy as np
import pandas as pd
import re


class AustralianDataset88_4gr:

    def __init__(self, data_root):
        self.groups = ["AD", "MCI", "NORM"]
        self.ds_root = data_root
        self.data_root = data_root

    def list_groups(self):
        return self.groups

    def list_subjects(self, group, kind):
        d = os.path.join(self.ds_root, kind, f"{group}-{kind}")
        return [subj for subj in os.listdir(d) if not subj.startswith(".")]

    def load_subject_sc(self, group, subj):
        kind = "SC"

        d = os.path.join(self.ds_root, kind, f"{group}-{kind}", subj)
        seperator = ''

        str_weights = os.path.join(d, 'Structural_matrix_'+subj+'.npz')
        str_wei = np.load(str_weights)
        weights = str_wei['weights']
        distance = str_wei['distance']

        return weights, distance

    def load_subject_ts(self, group, subj):
        kind = "TS"

        d = os.path.join(self.ds_root, kind, f"{group}-{kind}", subj)

        str_bold = os.path.join(d, 'Bold_'+subj+'.npz')
        Aud_TS = np.load(str_bold)

        Bold = Aud_TS['Bold']
        Aud_TS_corr_Z = Aud_TS['Aud_TS_corr_Z']
        Aud_TS_corr = Aud_TS['Aud_TS_corr']

        return Bold, Aud_TS_corr_Z, Aud_TS_corr

    def load_subject_region(self, group=None):

        ds_root_reg = self.data_root
        regions = open(os.path.join(ds_root_reg,  'healthy_3610A/Aud_88.txt'))

        reg_names = []
        for l in regions:
            as_list = l.split(" ")
            reg_names.append(as_list[0].replace("\n", ""))

        reg_names = reg_names[::2] + reg_names[1::2]
        return reg_names  # , reg_labels, reg_names

    def load_limbic(self, reg_names, group=None):

        lim_region = ['Hippocampus', 'ParaHippocampal', 'Cingulum',
                      'Amygdala', 'Temporal_Pole_Sup', 'Temporal_Mid']

        lim_vec = []
        lim_idxs = []

        for i in lim_region:

            lim = [region for region in reg_names if i in region]
            lim_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]

            lim_vec += [lim]
            lim_idxs += [lim_idx]

        lim_list = lim_vec[0]+lim_vec[1]+lim_vec[2] + \
            lim_vec[3]+lim_vec[4]+lim_vec[5]
        lim_idx = lim_idxs[0]+lim_idxs[1]+lim_idxs[2] + \
            lim_idxs[3]+lim_idxs[4]+lim_idxs[5]

        lim_left_idx = sorted(i for i in lim_idx if i < 44)
        lim_right_idx = sorted(i for i in lim_idx if i >= 44)

        lim_right_reg = [region for region in lim_list if '_R' in region]
        lim_left_reg = [region for region in lim_list if '_L' in region]

        lim_rsn = lim_right_reg + lim_left_reg
        lim_rsn_idx = lim_left_idx + lim_right_idx

        return lim_rsn, lim_rsn_idx, lim_left_reg, lim_right_reg, lim_left_idx, lim_right_idx

    def load_frontal(self, reg_names, group=None):

        frontal_region = ['Frontal', 'Precentral', 'Rolandic_Oper',
                          'Supp_Motor_Area', 'Olfactory', 'Paracentral_Lobule', 'Rectus']

        frontal_vec = []
        frontal_idxs = []

        for i in frontal_region:

            frontal = [region for region in reg_names if i in region]
        #     print(lim)
            frontal_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]

            frontal_vec += [frontal]
            frontal_idxs += [frontal_idx]

        frontal_list = frontal_vec[0]+frontal_vec[1]+frontal_vec[2] + \
            frontal_vec[3]+frontal_vec[4]+frontal_vec[5]+frontal_vec[6]
        frontal_idx = frontal_idxs[0]+frontal_idxs[1]+frontal_idxs[2] + \
            frontal_idxs[3]+frontal_idxs[4]+frontal_idxs[5]+frontal_idxs[6]

        frontal_left = sorted(i for i in frontal_idx if i < 44)
        frontal_right = sorted(i for i in frontal_idx if i >= 44)

        frontal_right_reg = [
            region for region in frontal_list if '_R' in region]
        frontal_left_reg = [
            region for region in frontal_list if '_L' in region]

        frontal_rsn = frontal_left_reg + frontal_right_reg
        frontal_rsn_idx = frontal_left + frontal_right

        return frontal_rsn, frontal_rsn_idx, frontal_left_reg, frontal_left, frontal_right_reg, frontal_right

    def load_temporal(self, reg_names, group=None):

        temp_region = ['Temporal', 'Hippocampus',
                       'ParaHippocampal', 'Fusiform', 'Amygdala']

        temp_vec = []
        temp_idxs = []

        for i in temp_region:

            temp = [region for region in reg_names if i in region]
            temp_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]
            temp_vec += [temp]
            temp_idxs += [temp_idx]

        temp_list = temp_vec[0]+temp_vec[1]+temp_vec[2]+temp_vec[3]+temp_vec[4]
        temp_idx = temp_idxs[0]+temp_idxs[1] + \
            temp_idxs[2]+temp_idxs[3]+temp_idxs[4]

        temp_left = sorted(i for i in temp_idx if i < 44)
        temp_right = sorted(i for i in temp_idx if i >= 44)

        temp_right_reg = [region for region in temp_list if '_R' in region]
        temp_left_reg = [region for region in temp_list if '_L' in region]

        temp_rsn = temp_left_reg + temp_right_reg
        temp_rsn_idx = temp_left + temp_right

        return temp_rsn, temp_rsn_idx, temp_left_reg, temp_left, temp_right_reg, temp_right

    def load_parietal(self, reg_names, group=None):

        parietal_region = ['Parietal', 'Postcentral',
                           'SupraMarginal', 'Angular', 'Precuneus']

        parietal_vec = []
        parietal_idxs = []

        for i in parietal_region:

            parietal = [region for region in reg_names if i in region]
        #     print(lim)
            parietal_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]
            parietal_vec += [parietal]
            parietal_idxs += [parietal_idx]

        parietal_list = parietal_vec[0]+parietal_vec[1] + \
            parietal_vec[2]+parietal_vec[3]+parietal_vec[4]
        parietal_idx = parietal_idxs[0]+parietal_idxs[1] + \
            parietal_idxs[2]+parietal_idxs[3]+parietal_idxs[4]

        parietal_left = sorted(i for i in parietal_idx if i < 44)
        parietal_right = sorted(i for i in parietal_idx if i >= 44)

        parietal_right_reg = [
            region for region in parietal_list if '_R' in region]
        parietal_left_reg = [
            region for region in parietal_list if '_L' in region]

        parietal_rsn = parietal_left_reg + parietal_right_reg
        parietal_rsn_idx = parietal_left + parietal_right

        return parietal_rsn, parietal_rsn_idx, parietal_left_reg, parietal_left, parietal_right_reg, parietal_right

    def load_occipital(self, reg_names, group=None):

        occipital_region = ['Occipital', 'Calcarine', 'Cuneus', 'Lingual']

        occipital_vec = []
        occipital_idxs = []

        for i in occipital_region:

            occipital = [region for region in reg_names if i in region]
        #     print(lim)
            occipital_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]
            occipital_vec += [occipital]
            occipital_idxs += [occipital_idx]

        occipital_list = occipital_vec[0] + \
            occipital_vec[1]+occipital_vec[2]+occipital_vec[3]
        occipital_idx = occipital_idxs[0]+occipital_idxs[1] + \
            occipital_idxs[2]+occipital_idxs[3]

        occipital_left = sorted(i for i in occipital_idx if i < 44)
        occipital_right = sorted(i for i in occipital_idx if i >= 44)

        occipital_right_reg = [
            region for region in occipital_list if '_R' in region]
        occipital_left_reg = [
            region for region in occipital_list if '_L' in region]

        occipital_rsn = occipital_left_reg + occipital_right_reg
        occipital_rsn_idx = occipital_left + occipital_right

        return occipital_rsn, occipital_rsn_idx, occipital_left_reg, occipital_left, occipital_right_reg, occipital_right

    def load_insulacingulate(self, reg_names, group=None):

        inscung_region = ['Insula', 'Cingulum']

        inscung_vec = []
        inscung_idxs = []

        for i in inscung_region:

            inscung = [region for region in reg_names if i in region]
        #     print(lim)
            inscung_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]
            inscung_vec += [inscung]
            inscung_idxs += [inscung_idx]

        inscung_list = inscung_vec[0]+inscung_vec[1]
        inscung_idx = inscung_idxs[0]+inscung_idxs[1]

        inscung_left = sorted(i for i in inscung_idx if i < 44)
        inscung_right = sorted(i for i in inscung_idx if i >= 44)

        inscung_right_reg = [
            region for region in inscung_list if '_R' in region]
        inscung_left_reg = [
            region for region in inscung_list if '_L' in region]

        inscung_rsn = inscung_left_reg + inscung_right_reg
        inscung_rsn_idx = inscung_left + inscung_right

        return inscung_rsn, inscung_rsn_idx, inscung_left_reg, inscung_left, inscung_right_reg, inscung_right

    def load_centralstructures(self, reg_names, group=None):

        centst_region = ['Caudate', 'Putamen', 'Pallidum', 'Thalamus']

        centst_vec = []
        centst_idxs = []

        for i in centst_region:

            centst = [region for region in reg_names if i in region]
        #     print(lim)
            centst_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]
            centst_vec += [centst]
            centst_idxs += [centst_idx]

        centst_list = centst_vec[0]+centst_vec[1]+centst_vec[2]+centst_vec[3]
        centst_idx = centst_idxs[0]+centst_idxs[1] + \
            centst_idxs[2]+centst_idxs[3]
        centst_left = sorted(i for i in centst_idx if i < 44)
        centst_right = sorted(i for i in centst_idx if i >= 44)

        centst_right_reg = [region for region in centst_list if '_R' in region]
        centst_left_reg = [region for region in centst_list if '_L' in region]

        centst_rsn_idx = centst_left + centst_right
        centst_rsn = centst_left_reg + centst_right_reg

        return centst_rsn, centst_rsn_idx, centst_left_reg, centst_left, centst_right_reg, centst_right

    def load_DMN(self, reg_names, group=None):

        dmn_region = ['Frontal_Mid', 'Frontal_Sup_Orb', 'Frontal_Med_Orb', 'Rectus', 'Cingulum_Ant', 'Cingulum_Post', 'Precuneus', 'Hippocampus', 'ParaHippocampal',
                      'Parietal_Inf', 'Angular', 'Temporal_Sup', 'Temporal_Pole_Sup', 'Temporal_Mid', 'Temporal_Inf']

        dmn_vec = []
        dmn_idxs = []

        for i in dmn_region:

            dmn = [region for region in reg_names if i in region]
            dmn_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]

            dmn_vec += [dmn]
            dmn_idxs += [dmn_idx]

        dmn_list = dmn_vec[0]+dmn_vec[1]+dmn_vec[2] + \
            dmn_vec[3]+dmn_vec[4]+dmn_vec[5]
        dmn_idx = dmn_idxs[0]+dmn_idxs[1]+dmn_idxs[2] + \
            dmn_idxs[3]+dmn_idxs[4]+dmn_idxs[5]

        dmn_left_idx = sorted(i for i in dmn_idx if i < 44)
        dmn_right_idx = sorted(i for i in dmn_idx if i >= 44)

        dmn_right_reg = [region for region in dmn_list if '_R' in region]
        dmn_left_reg = [region for region in dmn_list if '_L' in region]

        dmn_rsn = dmn_right_reg + dmn_left_reg
        dmn_rsn_idx = dmn_left_idx + dmn_right_idx

        return dmn_rsn, dmn_rsn_idx, dmn_left_reg, dmn_right_reg, dmn_right_idx, dmn_left_idx

    def load_cingulate(self, reg_names, group=None):

        lim_region = ['Cingulum']

        cin_vec = []
        cin_idxs = []

        for i in lim_region:

            cin = [region for region in reg_names if i in region]
            cin_idx = [idx for idx in range(
                len(reg_names)) if i in reg_names[idx]]

            cin_vec += [cin]
            cin_idxs += [cin_idx]

        cin_list = cin_vec[0]  # +lim_vec[1]+lim_vec[2]+lim_vec[3]
        cin_idx = cin_idxs[0]
        # +lim_idxs[1]+lim_idxs[2]+lim_idxs[3]

        cin_left_idx = sorted(i for i in cin_idx if i < 44)
        cin_right_idx = sorted(i for i in cin_idx if i >= 44)

        cin_right_reg = [region for region in cin_list if '_R' in region]
        cin_left_reg = [region for region in cin_list if '_L' in region]

        cin_rsn = cin_right_reg + cin_left_reg
        cin_rsn_idx = cin_left_idx + cin_right_idx

        return cin_rsn, cin_rsn_idx, cin_left_reg, cin_right_reg, cin_left_idx, cin_right_idx

    def New_Fluidity_sim_result(self, kind, group, list_, type_, test_):
        separator = ''

        ds_root = os.path.join(self.data_root, "interim",
                               "AUD_Simulation", "NEW_fluidity_sim_results")
        d = os.path.join(ds_root, kind)

        array = np.load(separator.join(
            [d, '/', group+'-MAX_FCD_VAR_'+list_+'_'+type_+'_fluidity_'+test_+'.npy']))

        return array

    def New_Fluidity_sim_result_samenoise(self, kind, group, list_, type_, test_):
        separator = ''

        ds_root = os.path.join(self.data_root, "interim",
                               "AUD_Simulation", "NEW_fluidity_sim_results")
        d = os.path.join(ds_root, kind)

        array = np.load(separator.join(
            [d, '/', group+'-MAX_FCD_VAR_'+list_+'_'+type_+'_fluidity_'+test_+'_SAME_NOISE.npy']))

        return array


def mask_M1(SC):
    """
    return interhemispheric links of SC
    """

    n, m = SC.shape
    assert(n == m)

    SC1 = np.zeros_like(SC)
    SC1[int(n/2):n, 0:int(n/2)] = SC[int(n/2):n, 0:int(n/2)]
    SC1[0:int(n/2), int(n/2):n] = SC[0:int(n/2), int(n/2):n]

    return SC1


def mask_M2(SC, region_indices):
    """
    return links of given regions of SC
    """

    n, m = SC.shape
    assert(n == m)

    SC1 = np.zeros_like(SC)

    left = []
    right = []

    for i in region_indices:
        if i < int(n/2):
            left.append(i)
        else:
            right.append(i)

    SC1[left, 0:int(n/2)] = SC[left, 0:int(n/2)]
    SC1[right, int(n/2):n] = SC[right, int(n/2):n]

    return SC1


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]


def set_k_diogonal(A, k, value=0.0):

    assert(len(A.shape) == 2)
    n = A.shape[0]
    assert(k < n)

    for i in range(-k, k+1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value

    return A
