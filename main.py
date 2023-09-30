# Databricks notebook source
# MAGIC %run ./read_data

# COMMAND ----------

# MAGIC %run ./preprocessing

# COMMAND ----------

# MAGIC %run ./training_model

# COMMAND ----------

# MAGIC %run ./return_embed

# COMMAND ----------

# MAGIC %run ./top_n

# COMMAND ----------

def load_model(model,column_name,vector_size,window_size):
  df = read_data()
  x = output_text(df,column_name)
  word2vec_model = model_train(x,vector_size,window_size,model)
  vectors = return_embed(word2vec_model,df,column_name)
  Vec = pd.DataFrame(vectors).transpose() # Saving vectors of each abstract in data frame so that we can use directly while running code again
  if model == 'Fasttext':
    Vec.to_csv('/dbfs/mnt/data/data/output/FastText_vec.csv')
  if model == 'Skipgram':
    Vec.to_csv('/dbfs/mnt/data/data/output/skipgram_vec.csv')

if __name__ == '__main__':
    # Load Word2Vec Skipgram model
    skipgram_model = load_model('Skipgram', 'Abstract', 100, 3)

    # Load FastText model
    fasttext_model = load_model('Fasttext', 'Abstract', 100, 3)    

# COMMAND ----------

# Get top similar results using Skipgram
query1 = 'Coronavirus'
results_skipgram, similarities_skipgram = top_n(query1, skipgram_model, 'Abstract')


# COMMAND ----------

# Get top similar results using FastText
query2 = 'Coronavirus'
results_fasttext, similarities_fasttext = top_n(query2, fasttext_model, 'Abstract')
