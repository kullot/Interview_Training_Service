# 관련 모듈 임포트
import numpy as np
import pandas as pd
import itertools
import random
from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os

# 함수 정의
def max_sum_sim(doc_embedding, candidate_embeddings, words, top_n, nr_candidates, candidates):
    # 문서와 각 키워드들 간의 유사도
    distances = cosine_similarity(doc_embedding, candidate_embeddings)

    # 각 키워드들 간의 유사도
    distances_candidates = cosine_similarity(candidate_embeddings, 
                                            candidate_embeddings)

    # 코사인 유사도에 기반하여 키워드들 중 상위 top_n개의 단어를 pick.
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # 각 키워드들 중에서 가장 덜 유사한 키워드들간의 조합을 계산
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def cos_sim(AVec, BVec):
  return np.dot(AVec, BVec) / ( np.linalg.norm(AVec) * np.linalg.norm(BVec) )


def check_answer_similar( userSentence='' ):
  if not userSentence:
    return '자기소개서를 입력해주세요.'
  # 1. 사용자의 입력 문자를 sbert 모델이 encode()를 호출하여 임베딩 한다 => 백터화
  embeddingSentence   = model.encode( userSentence )
  # 2. 데이터(챗봇 답변 컬럼(text))를 다 돌면서, 사용자가 입력한 문장과 유사도 계산해서 
  #    score 컬럼에 추가
  df_q['score'] = df_q.em.apply( lambda x: cos_sim( x, embeddingSentence) )
  # 3. 이중 최고 스코어를 받은 인덱스(idxmax)를 찾아서 문장을 리턴한다
  #    우연히 최고값이 2개 이상 나오면 가장 먼저 탐색된것이 추출된다
  # return df_q.loc[ df_q['score'].idxmax() ]['text'] # 유사도가 가장 높은 1가지 질문만 추출
  return df_q.loc[ random.choice(list(df_q['score'].nlargest(3).index)) ]['text']  # 유사도가 높은 3가지 질문 중 랜덤 택1

class Qbot:
  def __init__(self, intro):
    self.intro = intro

  # def listing_intro(intro):
  #   print("자기소개서를 입력해주세요.")
  #   intro_dict ={'intro' : intro}
  #   df_intro = pd.DataFrame(intro_dict, columns=['text'])
  #   df_intro[ 'em' ] = df_intro.text.apply( lambda x: model.encode(x) )
  #   docs = df_intro['text'].tolist()
  #   return docs

  def run_model(intro):
    intro_list = []
    result_list = []
    intro_list.append(intro)
    df_intro = pd.DataFrame(intro_list[0], columns=['text'])
    df_intro[ 'em' ] = df_intro.text.apply( lambda x: model.encode(x) )
    docs = df_intro['text'].tolist()
    for doc in docs:
      # 유사도 분석 범위
      n_gram_range = (2, 3)
      tokenized_doc = okt.pos(doc)
      tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

      count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
      candidates = count.get_feature_names_out()

      doc_embedding = model.encode([doc])
      candidate_embeddings = model.encode(candidates)

      keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n=10, nr_candidates=10, candidates=candidates)
      # print("="*100)
      # print("키워드 :", keywords)

      answer_list = []
      for input_intro in keywords:
        answer = check_answer_similar( input_intro )
        # print("질문 :", answer )
        answer_list.append(answer)
        
      df_answer = pd.DataFrame(answer_list)
      df_answer.columns = ['text']
      df_answer['em'] = df_answer.text.apply( lambda x: model.encode(x) )

      embeddingSentence   = model.encode( doc )
      df_answer['score'] = df_answer.em.apply( lambda x: cos_sim( x, embeddingSentence) )
      result = df_answer.loc[ list(df_answer['score'].nlargest(3).index) ]['text']
      # result = df_answer.loc[ df_answer['score'].idxmax() ]['text']
      # print("✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨")
      # print('최종 질문\n', result.unique())
      # print("✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨✨")  
      result_list.append(result)
    return result_list
    
  def make_qestions(result_list):
    questions = []
    for results in result_list:
      for result in results:
        questions.append(result)

    questions = list(set(questions))
    # q_dict = {'questions' : questions}
    return questions

def run_qbot(intro):
  # ques_list = []
  # for qus in intro:
  result_list = Qbot.run_model(intro)
  questions = Qbot.make_qestions(result_list)
    # questions = questions[0].replace(',','$')
    # print(questions, len(questions))
    # ques_list.append(questions)
  return questions

# intro ='항상 실현 가능하면서 구체적인 계획을 세우며 현실에 안주하지 않고 나 자신을 끊임없이 발전시키며 성장해나가는 것을 삶의 원동력으로 삼아왔습니다. 이러한 원동력을 바탕으로 20년 간 부동의 1위를 유지하는 롯데건설에서 근무하여 발전하고 싶습니다. 또한 미래의 자녀들이 살 공간에 기여하여 발전한 브랜드의 아파트에서 거주하길 바라는 마음에 롯데캐슬의 브랜드 파워를 증가하는데 기여하고 싶습니다.'
# ques = run_qbot(intro)
# print(ques)


# 모델 정의
okt = Okt()
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# 질문 데이터
# (출력데이터)
print(os.getcwd())
# df1 = pd.read_csv('//wsl.localhost/docker-desktop-data/data/docker/volumes/KangHee/final_Template/qbot/기업직무별_질문모음.csv', encoding='cp949', header=None)
# display(df1)
# df2 = pd.read_csv('//wsl.localhost/docker-desktop-data/data/docker/volumes/KangHee/final_Template/qbot/성격별질문모음.csv', encoding='cp949', header=None)
# display(df2)
# df1 = df1[[0]]
# df2 = df2[[0]]
# df1.columns = ['text']
# df2.columns = ['text']
# df_q = pd.concat([df1, df2], ignore_index=True)
df_q = pd.read_csv('C:/Users/mr1/Desktop/Final/final_Template/qbot/questions.csv', header=None)
df_q = df_q[[0]]
df_q.columns = ['text']


# # 질문데이터 em 파생변수 컬럼 생성
# # 모델 인코딩을 통한 벡터화
df_q[ 'em' ] = df_q.text.apply( lambda x: model.encode(x) )