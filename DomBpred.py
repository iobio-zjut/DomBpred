import os
import numpy as np
import time
import shutil


def classify_single_multi(msa_files):
    msa_pointer = open(msa_files, 'r')
    msa_list = msa_pointer.readlines()
    msa_pointer.close()

    eff_line_begin = eff_line_end = 0
    for i in range(0, len(msa_list)):
        if msa_list[i].isspace():
            eff_line_end += 1
            if eff_line_end == 2:
                eff_line_begin = i + 1
        if msa_list[i].startswith("#=GC RF"):
            eff_line_end = i
            break

    if eff_line_begin == 0:
        return -1
    else:
        eff_msa_begin = 0
        for i in range(0, len(msa_list[eff_line_begin])):
            if eff_msa_begin == 0:
                if (msa_list[eff_line_begin])[i].isspace():
                    eff_msa_begin = 1
            else:
                if not (msa_list[eff_line_begin])[i].isspace():
                    eff_msa_begin = i
                    break

        length = (len(msa_list[eff_line_begin]) - eff_msa_begin - 1)
        biaoqian_np = np.zeros((1, length), dtype=int)

        eff_line_begin += 1
        now_hang_s = (msa_list[eff_line_begin])[eff_msa_begin:-1]
        for i in range(0, len(now_hang_s)):
            if now_hang_s[i] != '-':
                biaoqian_np[0][i] = 1

        eff_line_begin += 1
        now_world_similarity = 0.0
        flag = False
        now_match_count = now_total_count = 0
        for i in range(eff_line_begin, eff_line_end):
            now_hang_s = msa_list[i]
            if now_hang_s.startswith('#'):
                continue

            now_hang_s = now_hang_s[eff_msa_begin:-1]

            flag = False
            for kk in range(0, biaoqian_np.shape[0]):
                now_match_count = now_total_count = 0

                for j in range(0, length):
                    if now_hang_s[j] != '-':
                        now_total_count += 1
                        if biaoqian_np[kk][j] != 0:
                            now_match_count += 1
                now_world_similarity = now_match_count / now_total_count

                if now_world_similarity >= 0.65:
                    for j in range(0, length):
                        if now_hang_s[j] != '-':
                            biaoqian_np[kk][j] = 1

                if now_world_similarity >= 0.5:
                    flag = True

            if (not flag) and (now_total_count >= 40):
                biaoqian_back_np = np.zeros((1, length), dtype=int)
                for j in range(0, length):
                    if now_hang_s[j] != '-':
                        biaoqian_back_np[0][j] = 1

                biaoqian_np = np.concatenate([biaoqian_np, biaoqian_back_np], axis=0)

        return biaoqian_np.shape[0]


def ising_son_update(pvalue_np, camark, seqlen):
    for resindex in range(0, seqlen):
        neighbor = 0.00
        for neig in range(0, seqlen):
            if pvalue_np[resindex][neig] != 0:
                if camark[resindex] < camark[neig]:
                    neighbor += pvalue_np[resindex][neig]
                if camark[resindex] > camark[neig]:
                    neighbor -= pvalue_np[resindex][neig]

        if neighbor > 0:
            camark[resindex] += 1
        if neighbor < 0:
            camark[resindex] -= 1

    return camark


def ising_cluster(dist_npz_files):
    dist_npz = np.load(dist_npz_files)

    distance_np = dist_npz['dist']
    seqlen = len(distance_np)

    pvalue_np = np.zeros([seqlen, seqlen], dtype=float)
    for ii in range(0, seqlen):
        for jj in range(0, seqlen):
            if ii != jj:
                if distance_np[ii][jj] < 14:
                    pvalue_np[ii][jj] = (14 / distance_np[ii][jj])

    camark = np.arange(0, seqlen, dtype=int)

    maxcycle = seqlen // 2
    stop_np = np.zeros([2, seqlen], dtype=float)
    stopflag_v = 0.00000
    for cycle in range(0, maxcycle):
        camark = ising_son_update(pvalue_np, camark, seqlen)

        if cycle == 0:
            for index in range(0, seqlen):
                stop_np[0, index] = camark[index]
        elif cycle == 1:
            for index in range(0, seqlen):
                stop_np[0, index] = (stop_np[0, index] + camark[index]) / 2
                stop_np[1, index] = camark[index]
        else:
            for index in range(0, seqlen):
                stop_np[1, index] = (stop_np[1, index] + camark[index]) / 2

            stopflag_v = 0.00000
            for ij in range(0, seqlen):
                temp_ave = (stop_np[0, ij] + stop_np[1, ij]) / 2
                temp_stdev = ((stop_np[0, ij] - temp_ave) ** 2) + ((stop_np[1, ij] - temp_ave) ** 2)
                stopflag_v += temp_stdev

            if stopflag_v < 0.000001:
                break

            for ij in range(0, seqlen):
                stop_np[0, ij] = stop_np[1, ij]
                stop_np[1, ij] = camark[ij]

    temporary_np = np.zeros(21, dtype=int)
    for index in range(0, seqlen):
        if index < 10:
            maxindex = (index + 10)
            local_np = np.zeros((maxindex + 1), dtype=int)
            median_index = (maxindex + 1) // 2

            for in_index in range(0, (maxindex + 1)):
                local_np[in_index] = camark[in_index]

            local_np = np.sort(local_np)
            camark[index] = local_np[median_index]
        elif index <= (seqlen - 11):
            temp_index = index - 10

            for in_index in range(0, 21):
                temporary_np[in_index] = camark[temp_index]
                temp_index += 1

            temporary_np = np.sort(temporary_np)
            camark[index] = temporary_np[10]
        else:
            min_index = (index - 10)
            locallength = (seqlen - min_index)
            local_np = np.zeros(locallength, dtype=int)
            median_index = locallength // 2

            for in_index in range(0, locallength):
                local_np[in_index] = camark[min_index]
                min_index += 1

            local_np = np.sort(local_np)
            camark[index] = local_np[median_index]

    return camark


def control_sub_dsc(contact_np, dot_list):
    conscore_v = 100000000.00
    splitdot_v = 0

    contact_length = len(contact_np)

    new_temp_dot_list = []
    for newi in range(0, len(dot_list)):
        if (dot_list[newi] >= 20) and (dot_list[newi] <= (contact_length - 20)):
            new_temp_dot_list.append(dot_list[newi])
    dot_list = new_temp_dot_list

    dot_length = len(dot_list)
    inner_score_np = np.zeros(dot_length, dtype=float)

    if dot_length < 1:
        return conscore_v, splitdot_v

    total_contact_v = 0
    before_d1_v = before_d2_v = 0

    domain_one = domain_two = inter_2domain = 0
    first_splitdot = dot_list[0]

    for i_index in range(0, contact_length):
        for j_index in range(0, contact_length):
            if contact_np[i_index][j_index] != 0:
                if i_index < first_splitdot:
                    if j_index < first_splitdot:
                        domain_one += 1
                    else:
                        inter_2domain += 1
                else:
                    if j_index >= first_splitdot:
                        domain_two += 1
                    else:
                        inter_2domain += 1
    if (domain_one == 0) or (domain_two == 0):
        inner_score_np[0] = -1.0
    else:
        temp_score_v = (2 * inter_2domain * (domain_one + domain_two)) / (domain_one * domain_two)
        inner_score_np[0] = temp_score_v


    total_contact_v = domain_one + domain_two + inter_2domain
    before_d1_v = domain_one
    before_d2_v = domain_two


    for index in range(1, dot_length):
        domain_one = before_d1_v
        domain_two = before_d2_v
        inter_2domain = 0

        for i_index in range(dot_list[index - 1], dot_list[index]):
            for j_index in range(0, contact_length):
                if contact_np[i_index][j_index] != 0:
                    if j_index < dot_list[index]:
                        domain_one += 1
                    if j_index >= dot_list[index - 1]:
                        domain_two -= 1

        for i_index in range(0, contact_length):
            for j_index in range(dot_list[index - 1], dot_list[index]):
                if contact_np[i_index][j_index] != 0:
                    if i_index < dot_list[index - 1]:
                        domain_one += 1
                    if i_index >= dot_list[index]:
                        domain_two -= 1

        if (domain_one == 0) or (domain_two == 0):
            inner_score_np[index] = -1.0
        else:
            inter_2domain = total_contact_v - domain_one - domain_two
            temp_score_v = (2 * inter_2domain * (domain_one + domain_two)) / (domain_one * domain_two)
            inner_score_np[index] = temp_score_v

        before_d1_v = domain_one
        before_d2_v = domain_two

    for iii in range(0, dot_length):
        if inner_score_np[iii] != (-1.0):
            if inner_score_np[iii] <= conscore_v:
                conscore_v = inner_score_np[iii]
                splitdot_v = dot_list[iii]

    return conscore_v, splitdot_v


def control_sub_dsd(contact_np, dot_list):
    disconscore_v = 100000000.00
    splitdot_1v = splitdot_2v = 0

    contact_length = len(contact_np)

    new_temp_dot_list = []
    for newi in range(0, len(dot_list)):
        if (dot_list[newi] >= 20) and (dot_list[newi] <= (contact_length - 20)):
            new_temp_dot_list.append(dot_list[newi])
    dot_list = new_temp_dot_list

    dot_length = len(dot_list)

    if dot_length < 2:
        return disconscore_v, splitdot_1v, splitdot_2v

    domain_1 = domain_2 = inter_domain12 = 0
    before_d1_v = before_d2_v = total_contact_v = 0

    for i_index in range(0, (dot_length - 1)):
        domain_1 = domain_2 = inter_domain12 = 0

        j_index = i_index + 1
        for iii in range(0, contact_length):
            for jjj in range(0, contact_length):
                if contact_np[iii][jjj] != 0:
                    if iii < dot_list[i_index]:
                        if jjj < dot_list[i_index]:
                            domain_1 += 1
                        elif jjj < dot_list[j_index]:
                            inter_domain12 += 1
                        else:
                            domain_1 += 1
                    elif iii < dot_list[j_index]:
                        if (jjj >= dot_list[i_index]) and (jjj < dot_list[j_index]):
                            domain_2 += 1
                        else:
                            inter_domain12 += 1
                    else:
                        if jjj < dot_list[i_index]:
                            domain_1 += 1
                        elif jjj < dot_list[j_index]:
                            inter_domain12 += 1
                        else:
                            domain_1 += 1

        if (domain_1 != 0) and (domain_2 != 0):
            temp_disscore_v = (2 * inter_domain12 * (domain_1 + domain_2)) / (domain_1 * domain_2)
            if temp_disscore_v < disconscore_v:
                disconscore_v = temp_disscore_v
                splitdot_1v = dot_list[i_index]
                splitdot_2v = dot_list[j_index]

        total_contact_v = domain_1 + domain_2 + inter_domain12
        before_d1_v = domain_1
        before_d2_v = domain_2

        for j_index in range((i_index + 2), dot_length):
            domain_1 = before_d1_v
            domain_2 = before_d2_v

            for iii in range(dot_list[j_index - 1], dot_list[j_index]):
                for jjj in range(0, contact_length):
                    if contact_np[iii][jjj] != 0:
                        if jjj < dot_list[i_index]:
                            domain_1 -= 1
                        elif jjj < dot_list[j_index]:
                            domain_2 += 1

                        if jjj >= dot_list[j_index - 1]:
                            domain_1 -= 1

            for iii in range(0, contact_length):
                for jjj in range(dot_list[j_index - 1], dot_list[j_index]):
                    if contact_np[iii][jjj] != 0:
                        if iii < dot_list[i_index]:
                            domain_1 -= 1
                        elif iii < dot_list[j_index - 1]:
                            domain_2 += 1

                        if iii >= dot_list[j_index]:
                            domain_1 -= 1

            if (domain_1 != 0) and (domain_2 != 0):
                inter_domain12 = total_contact_v - domain_1 - domain_2
                temp_disscore_v = (2 * inter_domain12 * (domain_1 + domain_2)) / (domain_1 * domain_2)
                if temp_disscore_v < disconscore_v:
                    disconscore_v = temp_disscore_v
                    splitdot_1v = dot_list[i_index]
                    splitdot_2v = dot_list[j_index]

            before_d1_v = domain_1
            before_d2_v = domain_2

    return disconscore_v, splitdot_1v, splitdot_2v


def oneto2c(contact_np, biaoqian_np, splitdot, dot_list):
    contact_length = len(contact_np)
    dot_len = len(dot_list)

    contact_len_1 = splitdot
    contact_len_2 = contact_length - splitdot

    contact_np_1 = np.zeros([contact_len_1, contact_len_1], dtype=int)
    contact_np_2 = np.zeros([contact_len_2, contact_len_2], dtype=int)

    biaoqian_1 = np.zeros(contact_len_1, dtype=int)
    biaoqian_2 = np.zeros(contact_len_2, dtype=int)

    dot_1_list = []
    dot_2_list = []

    for i in range(0, contact_length):
        for j in range(0, contact_length):
            if contact_np[i][j] != 0:
                if i < splitdot:
                    if j < splitdot:
                        contact_np_1[i][j] = 1
                else:
                    if j >= splitdot:
                        contact_np_2[(i - splitdot)][(j - splitdot)] = 1

    for ii in range(0, contact_length):
        if ii < splitdot:
            biaoqian_1[ii] = biaoqian_np[ii]
        else:
            biaoqian_2[(ii - splitdot)] = biaoqian_np[ii]

    for jj in range(0, dot_len):
        if dot_list[jj] < splitdot:
            dot_1_list.append(dot_list[jj])
        elif dot_list[jj] > splitdot:
            dot_2_list.append((dot_list[jj] - splitdot))

    return contact_np_1, contact_np_2, biaoqian_1, biaoqian_2, dot_1_list, dot_2_list


def oneto2d(contact_np, biaoqian_np, splitdot_1, splitdot_2, dot_list):
    contact_length = len(contact_np)
    dot_len = len(dot_list)

    contact_len_con = abs(splitdot_2 - splitdot_1)
    contact_len_discon = contact_length - contact_len_con

    contact_con_np = np.zeros([contact_len_con, contact_len_con], dtype=int)
    contact_discon_np = np.zeros([contact_len_discon, contact_len_discon], dtype=int)

    biaoqian_con = np.zeros(contact_len_con, dtype=int)
    biaoqian_discon = np.zeros(contact_len_discon, dtype=int)

    dot_1_con = []
    dot_2_discon = []

    for i in range(0, contact_length):
        for j in range(0, contact_length):
            if contact_np[i][j] != 0:
                if i < splitdot_1:
                    if j < splitdot_1:
                        contact_discon_np[i][j] = 1
                    elif j >= splitdot_2:
                        contact_discon_np[i][(j - contact_len_con)] = 1
                elif i < splitdot_2:
                    if (j >= splitdot_1) and (j < splitdot_2):
                        contact_con_np[(i - splitdot_1)][(j - splitdot_1)] = 1
                else:
                    if j < splitdot_1:
                        contact_discon_np[i - contact_len_con][j] = 1
                    elif j >= splitdot_2:
                        contact_discon_np[i - contact_len_con][j - contact_len_con] = 1

    for ii in range(0, contact_length):
        if ii < splitdot_1:
            biaoqian_discon[ii] = biaoqian_np[ii]
        elif ii < splitdot_2:
            biaoqian_con[(ii - splitdot_1)] = biaoqian_np[ii]
        else:
            biaoqian_discon[(ii - contact_len_con)] = biaoqian_np[ii]

    for jj in range(0, dot_len):
        if dot_list[jj] < splitdot_1:
            dot_2_discon.append(dot_list[jj])
        elif (dot_list[jj] > splitdot_1) and (dot_list[jj] < splitdot_2):
            dot_1_con.append((dot_list[jj] - splitdot_1))
        elif dot_list[jj] > splitdot_2:
            dot_2_discon.append((dot_list[jj] - contact_len_con))

    return contact_con_np, contact_discon_np, biaoqian_con, biaoqian_discon, dot_1_con, dot_2_discon


def recursion_son_adjust(ss2_file, old_dot_list):
    ss2_str = ""
    with open(ss2_file, 'r') as f:
        for line in f.readlines():
            line = (line.strip()).split()
            if len(line) == 6:
                ss2_str += line[2]

    coil_list = []
    flag = 0

    for i in range(0, len(ss2_str)):
        if ss2_str[i] == 'C' and flag == 0:
            flag = 1
            coil_list.append(i)
        elif ss2_str[i] != 'C' and flag == 1:
            flag = 0
            coil_list.append(i - 1)
    if flag == 1:
        coil_list.append((len(ss2_str) - 1))

    biaozhi_list = [0] * int(len(coil_list) / 2)
    for ii in range(0, len(old_dot_list)):
        for jj in range(1, len(coil_list), 2):
            if coil_list[jj] < old_dot_list[ii] < coil_list[jj + 1]:
                left_diff_v = old_dot_list[ii] - coil_list[jj]
                right_diff_v = coil_list[jj + 1] - old_dot_list[ii]
                if left_diff_v <= right_diff_v:
                    biaozhi_list[int((jj - 1) / 2)] = 1
                else:
                    biaozhi_list[int((jj + 1) / 2)] = 1
                break

    final_dot_list = old_dot_list
    for jj in range(0, len(biaozhi_list)):
        if biaozhi_list[jj] == 1:
            begin_v = coil_list[(jj * 2)]
            end_v = coil_list[(jj * 2 + 1)]
            for ii in range(begin_v, end_v + 1):
                final_dot_list.append(ii)

    final_dot_list = list(set(final_dot_list))
    final_dot_list.sort()

    return final_dot_list


def recursion_son_control(contact_np, biaoqian_np, dot_list, result_str_list):
    cut2c_v = 0.56
    cut2d_v = 0.45

    continues_score = control_sub_dsc(contact_np, dot_list)
    discontinues_score = control_sub_dsd(contact_np, dot_list)

    if continues_score[0] < cut2c_v:
        if discontinues_score[0] < cut2d_v:
            gap_continues = cut2c_v - continues_score[0]
            gap_discontinues = cut2d_v - discontinues_score[0]
            if gap_continues >= gap_discontinues:
                temp_result = oneto2c(contact_np, biaoqian_np, continues_score[1], dot_list)
                recursion_son_control(temp_result[0], temp_result[2], temp_result[4], result_str_list)
                recursion_son_control(temp_result[1], temp_result[3], temp_result[5], result_str_list)
            else:
                temp_result = oneto2d(contact_np, biaoqian_np, discontinues_score[1], discontinues_score[2], dot_list)
                recursion_son_control(temp_result[0], temp_result[2], temp_result[4], result_str_list)
                recursion_son_control(temp_result[1], temp_result[3], temp_result[5], result_str_list)
        else:
            temp_result = oneto2c(contact_np, biaoqian_np, continues_score[1], dot_list)
            recursion_son_control(temp_result[0], temp_result[2], temp_result[4], result_str_list)
            recursion_son_control(temp_result[1], temp_result[3], temp_result[5], result_str_list)
    else:
        if discontinues_score[0] < cut2d_v:
            temp_result = oneto2d(contact_np, biaoqian_np, discontinues_score[1], discontinues_score[2], dot_list)
            recursion_son_control(temp_result[0], temp_result[2], temp_result[4], result_str_list)
            recursion_son_control(temp_result[1], temp_result[3], temp_result[5], result_str_list)
        else:
            strings = str(biaoqian_np[0]) + "-"

            biaoqian_len_v = len(biaoqian_np) - 1
            for iji in range(1, biaoqian_len_v):
                if biaoqian_np[iji] != (biaoqian_np[iji - 1] + 1):
                    strings = strings + str(biaoqian_np[iji - 1]) + "," + str(biaoqian_np[iji]) + "-"

            strings = strings + str(biaoqian_np[biaoqian_len_v]) + ";"
            result_str_list.append(strings)
            return
    pass


def recursion(npz_str_name_info, dist_file_info, camark_np, ss2_file_info):
    dist_npz = np.load(dist_file_info)
    distance_np = dist_npz['dist']

    seqlen = len(distance_np)
    contact_np = np.zeros([seqlen, seqlen], dtype=int)
    for i_index in range(0, (seqlen - 1)):
        for j_index in range((i_index + 1), seqlen):
            if distance_np[i_index][j_index] <= 8:
                contact_np[i_index][j_index] = 1
                contact_np[j_index][i_index] = contact_np[i_index][j_index]

    biaoqian_np = np.arange(1, (seqlen + 1), dtype=int)

    dot_list = []
    for jij in range(6, (seqlen - 6)):
        value = abs(camark_np[jij + 1] - camark_np[jij])
        if value >= 1:
            dot_list.append(jij)
    dot_list = recursion_son_adjust(ss2_file_info, dot_list)

    result_str_list = []
    recursion_son_control(contact_np, biaoqian_np, dot_list, result_str_list)

    pred_str = npz_str_name_info + "  " + str(len(result_str_list)) + "  "
    for jij in range(0, len(result_str_list)):
        pred_str = pred_str + result_str_list[jij]

    return pred_str


if __name__ == '__main__':
    begin_time = time.time()

    input_seq_file = r""
    with open(input_seq_file, 'r') as f:
        alls = f.readlines()
        input_seq_name = (alls[0].strip())[1:]
        input_seq_length = len(alls[1].strip())

    # 从SDSL中使用Jackhmmer搜索MSA，再进行单域多域初步判断
    MSA_file = r""

    result_a = classify_single_multi(MSA_file)
    if result_a == 1:
        print(input_seq_name, "  1  ", "1-", input_seq_length, ';')
    else:
        # 若初步判断为多域蛋白，则继续后面的过程
        # 该距离图来自trRosetta，原类型为L*L*37，现使用的类型为L*L*1
        DIST_file = r""
        camark = ising_cluster(DIST_file)

        # 当聚类完毕后，开始启动分析
        # 该二级结构信息来自PSIPRED
        SS2_file = r""
        result = recursion(input_seq_name, DIST_file, camark, SS2_file)
        print(result)

    end_time = time.time()
    print("The elapsed time is %f seconds" % (end_time - begin_time))

    # 单步调试过程
    # for files in os.listdir(r"C:\Users\yzz\Desktop\sprint\单多域识别\result_single_1004"):
    #     fild=r"C:\Users\yzz\Desktop\sprint\单多域识别\result_single_1004"+'/'+files
    #     resul=classify_single_multi(fild)
    #     print(resul)

    pass
