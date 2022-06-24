import pandas as pd
import numpy as np

new_annot_dir = 'E:/Felix/1822041/Tugas Akhir/annot/annot_binary_final.csv'
annotate = pd.read_csv(new_annot_dir)
print('Annotation shape: ', annotate.shape)
df_imf_1_dir = 'E:/Felix/1822041/Tugas Akhir/new/imf_1/record_imf1_a19_s8.csv'

'concat with'
df2_imf_1_dir= 'E:/Felix/1822041/Tugas Akhir/new/imf_1/record_imf1_c10_s7.csv'


'IMF1' 
df_imf_1 = pd.read_csv(df_imf_1_dir)
print('Success reading df_imf1_a01_b05 from : ', df_imf_1_dir,' with shape: ', df_imf_1.shape)

df2_imf_1= pd.read_csv(df2_imf_1_dir)
print('Success reading df_imf1_c10 from : ', df2_imf_1_dir,' with shape: ', df2_imf_1.shape)

df_concat_imf1 = pd.DataFrame([])
df_concat_imf1 = df_concat_imf1.append(df_imf_1).append(df2_imf_1)#concat files
save_df_concat_imf1 = 'E:/Felix/1822041/Tugas Akhir/new/imf_1/record_imf1_final.csv'

print('Concat success... with shape: ', df_concat_imf1.shape)
print('Saving concated files to ', save_df_concat_imf1)
df_concat_imf1.to_csv(save_df_concat_imf1,index=False)
print('Success saving df_concat_imf1 files to: ', save_df_concat_imf1, ' with shape: ', df_concat_imf1.shape)

'Annot Mapping'
annot_map = annotate.to_numpy()
annot_map = np.where(annot_map == 'N', 0,annot_map)
annot_map = np.where(annot_map == 'A', 1,annot_map)

annot_map = np.array(annot_map)
annot_map = pd.DataFrame(annot_map)
annot_map.to_csv('D:/Kuliyah/.Tugas Akhir/cleaned data/annot/test/annotNew_c10_s7.csv',index=False)