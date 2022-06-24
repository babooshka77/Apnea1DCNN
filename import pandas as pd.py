import pandas as pd
import numpy as np


new_annot_dir = 'E:/Felix/1822041/Tugas Akhir/annot/annot_binary_final.csv'
annotate = pd.read_csv(new_annot_dir)
print('Annotation shape: ', annotate.shape)

'Data Final dgn array (15600,600)'
save_df_concat_imf1 = 'E:/Felix/1822041/Tugas Akhir/new/imf_1/record_imf1_final.csv'

'Menggabungkan Annot dengan Data Final'
df_final_concat_imf_1 = pd.read_csv(save_df_concat_imf1)
'Melettakan data Annot pada kolom terakhir'
df_final_concat_annot_imf_1 = pd.concat([df_final_concat_imf_1, annotate],axis=1)
print('ANNOT Concat  success... with shape: ', df_final_concat_annot_imf_1.shape)
save_df_final_concat_annot_imf_1 = 'E:/Felix/1822041/Tugas Akhir/new/imf_1/recordAnnot_imf1_final.csv'
df_final_concat_annot_imf_1.to_csv(save_df_final_concat_annot_imf_1, index=False)
print('Success saving df_final_concat_imf_1 files to: ', save_df_final_concat_annot_imf_1, ' with shape: ', df_final_concat_annot_imf_1.shape)

df_imf_1_concat = pd.read_csv(save_df_final_concat_annot_imf_1)
df_imf_1_concat_sort = df_imf_1_concat.sort_values(by=df_imf_1_concat.columns[6000])
df_imf_1_concat_sort = df_imf_1_concat_sort.reset_index(drop=True)
df_imf_1_potong = df_imf_1_concat_sort.iloc[3470:]
print('Success balancing df_imf_1_potong with shape:',df_imf_1_potong.shape)
df_imf_1_potong.to_csv(save_df_imf_1_potong, index=False)
print('df_imf_1_potong saved to ', save_df_imf_1_potong)


print('Time elapsed: ', timedelta(seconds=end-start))
