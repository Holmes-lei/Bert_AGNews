import tensorflow as tf
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 超参数设置
max_length = 32
# 在用kaggle的GPU进行运行的时候batch_size绝对不能超过
batch_size = 32

from sklearn.model_selection import train_test_split
import pandas as pd

def split_dataset(df):
    train_set, val_set = train_test_split(df, 
        stratify=df['label'],
        test_size=0.1,
        random_state=42)

    return train_set,val_set

# df_raw = pd.read_csv("train.csv",nrows =20,encoding='utf-8',header=None,names=["y","title","text"]) # 截取前20行
df_raw_train = pd.read_csv("./data/train.csv",encoding='utf-8',header=None,names=["y","title","text"])
df_raw_train["text"] = df_raw_train["title"].str.cat(df_raw_train["text"],sep=' ') # 将SGNews数据集的Title和Context都连接起来
print(df_raw_train.head())

# label
df_label = pd.DataFrame({"label":["World","Sports","Business","Tech"],"y":list(range(0, 4))})  # range(1,5)包含了[1，2，3，4]
print(df_label)
df_label["y"] = df_label["y"].astype(int) # 将生成的label映射矩阵的y列中的全部值也就是[1,2,3,4]转换成int类型

''' 
    由于输入到Bert模型中的labels，
    也就是进行model训练的时候，
    选择的参数labels有多少，
    默认是从0开始的，
    因此这里要对原始数据的label做一个映射，
    将原先的label都映射到新的值上。
'''
# https://stackoverflow.com/questions/64495230/how-to-handle-the-invalid-argument-error-for-output-labels-in-tf-keras/64495478#64495478
# map_dict = {1:0, 2:1, 3:2, 4:3}

# for k, v in map_dict.items():
#     df_raw.loc[df_raw["y"]==k, df_raw["y"]] = v
# print(df_raw["y"])

df_raw_train["y"] = df_raw_train["y"].astype(int) # 将原始数据的y列中的全部值也就是原始数据集对应的label的序号转换成int类型，看前面载入数据时，将label序号那一列的名字起为了y

for index in df_raw_train.index:
    df_raw_train.loc[index, "y"] = df_raw_train.loc[index, "y"] - 1
        
df_raw_train = pd.merge(df_raw_train,df_label,on="y",how="left") # 第一个和第二个参数表示哪两个表进行连接，on参数确定哪个字段作为主键，how参数表示左连接

print(df_raw_train)

train_data,val_data = split_dataset(df_raw_train)

df_raw_test = pd.read_csv("./data/test.csv",encoding='utf-8',header=None,names=["y","title","text"])
df_raw_test["text"] = df_raw_test["title"].str.cat(df_raw_test["text"],sep=' ') # 将SGNews数据集的Title和Context都连接起来

df_raw_test["y"] = df_raw_test["y"].astype(int) # 将原始数据的y列中的全部值也就是原始数据集对应的label的序号转换成int类型，看前面载入数据时，将label序号那一列的名字起为了y

for index in df_raw_test.index:
    df_raw_test.loc[index, "y"] = df_raw_test.loc[index, "y"] - 1
        
test_data = pd.merge(df_raw_test,df_label,on="y",how="left") # 第一个和第二个参数表示哪两个表进行连接，on参数确定哪个字段作为主键，how参数表示左连接

print(df_raw_train)

def convert_example_to_feature(review):
    return tokenizer.encode_plus(review, 
                                 add_special_tokens = True, # add [CLS], [SEP]
                                 max_length = max_length, # max length of the text that can go to BERT
                                 pad_to_max_length = True, # add [PAD] tokens
                                 return_attention_mask = True, # add attention mask to not focus on pad tokens
                                )

# map to the expected input to TFBertForSequenceClassification, see here 
def map_example_to_dict(input_ids, attention_masks, token_type_ids, label):
    return {
      "input_ids": input_ids,
      "token_type_ids": token_type_ids,
      "attention_mask": attention_masks,
  }, label

def encode_examples(ds, limit=-1):
    # prepare list, so that we can build up final TensorFlow dataset from slices.
    input_ids_list = []
    token_type_ids_list = []
    attention_mask_list = []
    label_list = []
    if (limit > 0):
        ds = ds.take(limit)
    
    for index, row in ds.iterrows():
        review = row["text"]
        label = row["y"]
        bert_input = convert_example_to_feature(review)
  
        input_ids_list.append(bert_input['input_ids'])
        token_type_ids_list.append(bert_input['token_type_ids'])
        attention_mask_list.append(bert_input['attention_mask'])
        label_list.append([label])
    return tf.data.Dataset.from_tensor_slices((input_ids_list, attention_mask_list, token_type_ids_list, label_list)).map(map_example_to_dict)

# https://blog.csdn.net/weixin_48344945/article/details/110479978
# train dataset
ds_train_encoded = encode_examples(train_data).shuffle(10000).batch(batch_size)
# val dataset
ds_val_encoded = encode_examples(val_data).batch(batch_size)
# test dataset
ds_test_encoded = encode_examples(test_data).batch(batch_size)

from transformers import TFBertForSequenceClassification
import tensorflow as tf

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)



# recommended learning rate for Adam 5e-5, 3e-5, 2e-5
# learning_rate = 2e-5
learning_rate = 3e-5
# we will do just 1 epoch for illustration, though multiple epochs might be better as long as we will not overfit the model
number_of_epochs = 2

# evaluate whether GPU is available
print(tf.test.is_gpu_available())

# model initialization
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# optimizer Adam recommended
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)

# we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

model.summary()
# fit model
bert_history = model.fit(ds_train_encoded, epochs=number_of_epochs, validation_data=ds_val_encoded)
# evaluate test set
model.evaluate(ds_test_encoded)
