import pandas as pd
from pep_modules import patientai as pai

df_new_seq = pd.read_pickle(r'../data/pat_journey_rx_px_dx.pkl')

patai_obj = pai(df_new_seq[['pat_id','ICD9_CODE', 'seq_num_final','switch_flag','cohort']]
                ,'pat_id','ICD9_CODE', 'seq_num_final','switch_flag','cohort')

cross_df = patai_obj.get_crossectional_data(30)

targets_df = patai_obj.get_target()

cross_df.to_pickle(r'../data/cross_data.pkl')

targets_df.to_pickle(r'../data/targets.pkl')

print("Crossectional data generated!")
