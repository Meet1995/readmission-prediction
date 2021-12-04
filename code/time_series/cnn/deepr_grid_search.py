import pickle
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec
from tensorflow.keras import callbacks
from sklearn.model_selection import train_test_split

from pep_modules import patientai as pai


cross = pd.read_pickle(r'../data/cross_data.pkl')
target = pd.read_pickle(r'../data/target.pkl')

X1, X_test, y1, y_test = train_test_split(cross, target, test_size=0.20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X1, y1, test_size=0.15, random_state=42)

x_train_static = pd.read_pickle(r'../data/train_static.pkl')
x_val_static = pd.read_pickle(r'../data/val_static.pkl')
x_test_static = pd.read_pickle(r'../data/test_static.pkl')

para_dict = {
    'w2v_dim':[],
    'journey_len':[],
    'window_size':[],
    'train_auc':[],
    'val_auc':[],
    'test_auc':[]
}

static_para_dict = {
    'w2v_dim':[],
    'journey_len':[],
    'window_size':[],
    'train_auc':[],
    'val_auc':[],
    'test_auc':[]
}

for dim in [32, 64, 128]:
    print("Word2Vec: ", dim)
    
    loaded_word2vec = Word2Vec.load(f'../saved_models/word2vec_{dim}.model')
    for j_len in [50, 85, 120]:
        print("Journey Length: ", j_len)
        
        x_train_tensor, y_train_tensor = pai.get_all_tensors_deepr(X_train,'journey',
                                                                         y_train,'switch_flag',loaded_word2vec,j_len)
        x_val_tensor, y_val_tensor = pai.get_all_tensors_deepr(X_val,'journey',
                                                                     y_val,'switch_flag',loaded_word2vec,j_len)
        x_test_tensor, y_test_tensor = pai.get_all_tensors_deepr(X_test,'journey',
                                                                       y_test,'switch_flag',loaded_word2vec, j_len)
        
        for win in [3,5,6]:
            print("Window Size: ", win)
            
            para_dict['w2v_dim'].append(dim)
            static_para_dict['w2v_dim'].append(dim)
            para_dict['journey_len'].append(j_len)
            static_para_dict['journey_len'].append(j_len)
            para_dict['window_size'].append(win)
            static_para_dict['window_size'].append(win)
            
            es_callback = callbacks.EarlyStopping(monitor='val_my_auc', patience=7, mode='max',restore_best_weights=True)
            model = pai.build_model_deepr(input_shape=(loaded_word2vec.vector_size,j_len,1), kernel_size=(loaded_word2vec.vector_size,win))
            model.fit(x_train_tensor,y_train_tensor, epochs=100, callbacks=[es_callback], validation_data=(x_val_tensor,y_val_tensor))
            model.save_weights(f'../saved_models/deepr_tuning/deepr_w2v{dim}_jlen{j_len}_win{win}.h5')
            
            para_dict['train_auc'].append(model.evaluate(x_train_tensor,y_train_tensor)[1])
            para_dict['val_auc'].append(model.evaluate(x_val_tensor,y_val_tensor)[1])
            para_dict['test_auc'].append(model.evaluate(x_test_tensor,y_test_tensor)[1])
            
            
            static_callback = callbacks.EarlyStopping(monitor='val_my_auc', patience=7, mode='max',restore_best_weights=True)
            static_model = pai.build_model_deepr(input_shape=(loaded_word2vec.vector_size,j_len,1), 
                                                 kernel_size=(loaded_word2vec.vector_size,win), n_static=182)
            static_model.fit([x_train_tensor, x_train_static],y_train_tensor, epochs=100, callbacks=[static_callback], 
                              validation_data=([x_val_tensor,x_val_static],y_val_tensor))
            static_model.save_weights(f'../saved_models/deepr_tuning/static_deepr_w2v{dim}_jlen{j_len}_win{win}.h5')
            
            static_para_dict['train_auc'].append(static_model.evaluate([x_train_tensor, x_train_static],y_train_tensor)[1])
            static_para_dict['val_auc'].append(static_model.evaluate([x_val_tensor,x_val_static],y_val_tensor)[1])
            static_para_dict['test_auc'].append(static_model.evaluate([x_test_tensor,x_test_static],y_test_tensor)[1])
            

with open(f'../saved_models/deepr_tuning/para_dict.pkl', 'wb') as f:
    pickle.dump(para_dict, f)

with open(f'../saved_models/deepr_tuning/static_para_dict.pkl', 'wb') as f:
    pickle.dump(static_para_dict, f)

print("Grid search done! All results dumped to home_folder - saved_models - deepr_tuning.")
    