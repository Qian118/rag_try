import sys
from catboost import CatBoostClassifier, CatBoostRegressor
from imblearn.over_sampling import RandomOverSampler
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, GroupKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import xgboost as xgb
from xgboost import plot_importance
from ydata_profiling import ProfileReport

pd.set_option('display.max_columns', None)
sns.set_palette(sns.color_palette("Set2"))

print(f"Running on {sys.version}")

GENERATE_REPORTS = False

KAGGLE_ENV = True
if KAGGLE_ENV:
    data_path = "../input/"
else:
    data_path = ""
# 读取train数据集，并删除id列
df_train = pd.read_csv(data_path + "playground-series-s3e26/train.csv").drop(
    ["id"], axis=1
)
# 读取test数据集
df_test = pd.read_csv(data_path + "playground-series-s3e26/test.csv")
# 取出id列
test_IDs = df_test.id
# 删除test数据集中的id列
df_test = df_test.drop("id", axis=1)
# 规整顺序
df_test = df_test[df_train.columns[:-1]]
# 读取提交数据集
df_sample_sub = pd.read_csv(data_path + "playground-series-s3e26/sample_submission.csv")

# 读取父类数据集，并使用columns使其与train数据集特征顺序保持一致
df_supp = pd.read_csv(
    data_path + "cirrhosis-patient-survival-prediction/cirrhosis.csv"
)[df_train.columns]

# 合并train数据集和supp父类数据集，并重置索引
df_train = pd.concat(objs=[df_train, df_supp]).reset_index(drop=True)

# 目标列
LABEL = "Status"
# 字符型特征
CAT_FEATS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
# 筛选出数值型特征
NUM_FEATS = [x for x in df_train.columns if x not in CAT_FEATS and x != LABEL]
# 原始特征（除了标签列以外的所有其他列）
ORG_FEATS = df_train.drop(LABEL, axis=1).columns.tolist()

print(f"Train shape: {df_train.shape}")
print(f"Test shape: {df_test.shape}")
# 对数据进行统计分析

# 对训练集中所有的列进行描述性统计（包括字符类型的特征）
desc_df = df_train.describe(include="all")
# 对描述性统计列表进行转置
desc_df = desc_df.T
# nunique：统计每一列的非空唯一值的数量
# fillna：对desc_df['unique']中的缺失值进行填补
desc_df['unique'] = desc_df['unique'].fillna(df_train.nunique())
# 把样本计数（非空样本）列转换为int格式
desc_df['count'] = desc_df['count'].astype('int16')
# 总样本数 - 非空样本数 = 缺失样本数
desc_df['missing'] = df_train.shape[0] - desc_df['count']
desc_df
if GENERATE_REPORTS:
    profile = ProfileReport(df_train, title="YData Profiling Report - Cirrhosis")
    profile.to_notebook_iframe()
# 对标签列的每一类样本个数进行计数
status_counts = df_train[LABEL].value_counts()
# 取出标签
labels = status_counts.index
# 取出对应数值
sizes = status_counts.values
# 计算每一类的占比
percentages = 100.0 * (sizes/sizes.sum())

# 新建画布并规定大小
plt.figure(figsize=(10, 6))
# 画饼图
plt.pie(sizes, labels=[f"{l}, {s:.1f}%" for l, s in zip(labels, percentages)], startangle=90)
# 环境设置
plt.gca().set_aspect("equal")
# 图例
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1), labels=labels, title=LABEL)
# 标题
plt.title(f"Distribution of {LABEL}")
plt.show()
# 规定画布的大小
plt.figure(figsize=(14, len(CAT_FEATS) * 2))
for i, col in enumerate(CAT_FEATS):
    # 绘制子图
    plt.subplot(len(CAT_FEATS) // 2 + 1, 3, i + 1)
    sns.countplot(x=col, hue=LABEL, data=df_train)
    plt.title(f"{col} vs {LABEL}")
    plt.tight_layout()
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, ax in enumerate(axes.flatten()):
    sns.violinplot(x=LABEL, y=NUM_FEATS[i], data=df_train, ax=ax)
    ax.set_title(f"{NUM_FEATS[i]} vs {LABEL}")
plt.tight_layout()
plt.show()
DATA_VERSION = 24

CAT_FEATS = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema", "Stage"]
NUM_FEATS = [x for x in df_train.columns if x not in CAT_FEATS and x != LABEL]

# 对训练集和测试集进行深拷贝
df_train_mod = df_train.copy(deep=True)
df_test_mod = df_test.copy(deep=True)

print(f"Train shape: {df_train_mod.shape}")
print(f"Test shape: {df_test_mod.shape}")
# 带有缺失值并且数据类型为objct的列
missing_cat=[f for f in df_train_mod.columns if df_train_mod[f].dtype=="O" if df_train_mod[f].isna().sum()>0]

cat_params={
            'depth': 6,
            'learning_rate': 0.1,
            'l2_leaf_reg': 0.7,
            'random_strength': 0.2,
            'max_bin': 200,
            'od_wait': 65,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'Iter',
            'eval_metric': 'MultiClass',
            'loss_function': 'MultiClass',
}

def store_missing_rows(df: pd.Dataframe, features: List[str]) -> Dict[str, pd.Dataframe]:
    # 初始化字典
    missing_rows = {}
    
    # 遍历特征
    for feature in features:
        # 取当前特征下存在缺失的样本
        missing_rows[feature] = df[df[feature].isnull()]
    
    return missing_rows

def fill_missing_categorical(train, test, target, features, max_iterations=10):
    # 将去掉目标列的训练集与测试集进行拼接
    df = pd.concat([train.drop(columns=target), test], axis="rows")
    # 重置索引
    df = df.reset_index(drop=True)
    
    # Dict['str','pd.Dataframe']
    missing_rows = store_missing_rows(df, features)
    
    # 标记缺失样本
    for f in features:
        df[f] = df[f].fillna("Missing_" + f)

    
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in features:
            # 取当前特征下存在缺失的样本的行索引
            rows_miss = missing_rows[feature].index
            # 根据行索引取出缺失样本并进行深拷贝
            missing_temp = df.loc[rows_miss].copy()
            # 从训练数据中删除缺失的样本并进行深拷贝
            non_missing_temp = df.drop(index=rows_miss).copy()
            # 删除feature列（feature全为NaN）
            missing_temp = missing_temp.drop(columns=[feature])
            # 筛选出不是featur列并且为ojbect类型的列
            other_features = [x for x in df.columns if (x != feature and df[x].dtype == "O")]
            
            # 删除未缺失的数据集中的feature（也可以称为label）
            X_train = non_missing_temp.drop(columns=[feature])
            # 取出未缺失数据集中的label
            y_train = non_missing_temp[[feature]]
            
            # 初始化Catboost模型
            catboost_classifier = CatBoostClassifier(**cat_params)
            catboost_classifier.fit(X_train, y_train, cat_features=other_features, verbose=False)
            
            # Catboost预测过程
            y_pred = catboost_classifier.predict(missing_temp)
            
            if y_pred.dtype != "O":
                y_pred = y_pred.astype(str)

            df.loc[rows_miss, feature] = y_pred

    train[features] = np.array(df.iloc[:train.shape[0]][features])
    test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train, test
missing_num=[f for f in df_train_mod.columns if df_train_mod[f].dtype!="O" and df_train_mod[f].isna().sum()>0]

cb_params = {
            'iterations': 500,
            'depth': 6,
            'learning_rate': 0.02,
            'l2_leaf_reg': 0.5,
            'random_strength': 0.2,
            'max_bin': 150,
            'od_wait': 80,
            'one_hot_max_size': 70,
            'grow_policy': 'Depthwise',
            'bootstrap_type': 'Bayesian',
            'od_type': 'IncToDec',
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',
            'random_state': 42,
        }
lgb_params = {
            'n_estimators': 50,
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.20,
            'colsample_bytree': 0.56,
            'reg_alpha': 0.25,
            'reg_lambda': 5e-08,
            'objective': 'multiclass',
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'random_state': 42,
        }

def rmse(y1,y2):
    return(np.sqrt(mean_squared_error(y1,y2)))

def fill_missing_numerical(train,test,target, features, max_iterations=10):
    train_temp=train.copy()
    if target in train_temp.columns:
        train_temp=train_temp.drop(columns=target)
    df=pd.concat([train_temp,test],axis="rows")
    df=df.reset_index(drop=True)
    missing_rows = store_missing_rows(df, features)
    
    for f in features:
        # 使用当前列的均值填补缺失值
        df[f]=df[f].fillna(df[f].mean())
    
    cat_features=[f for f in df.columns if not pd.api.types.is_numeric_dtype(df[f])]
    dictionary = {feature: [] for feature in features}
    
    for iteration in tqdm(range(max_iterations), desc="Iterations"):
        for feature in features:
            rows_miss = missing_rows[feature].index
            
            missing_temp = df.loc[rows_miss].copy()
            non_missing_temp = df.drop(index=rows_miss).copy()
            y_pred_prev=missing_temp[feature]
            missing_temp = missing_temp.drop(columns=[feature])

            X_train = non_missing_temp.drop(columns=[feature])
            y_train = non_missing_temp[[feature]]
            
            model = CatBoostRegressor(**cb_params)
#             if iteration>3:
#                 model = lgb.LGBMRegressor(**lgb_params)
            model.fit(X_train, y_train,cat_features=cat_features, verbose=False)
            
            y_pred = model.predict(missing_temp)
            df.loc[rows_miss, feature] = y_pred
            error_minimize=rmse(y_pred, y_pred_prev)
            dictionary[feature].append(error_minimize)

    for feature, values in dictionary.items():
        iterations = range(1, len(values) + 1) 
        plt.plot(iterations, values, label=feature)
        plt.xlabel('Iterations')
        plt.ylabel('RMSE')
        plt.title('Minimization of RMSE with iterations')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    train[features] = np.array(df.iloc[:train.shape[0]][features])
    test[features] = np.array(df.iloc[train.shape[0]:][features])

    return train,test
DROP_MISSING = True

if DROP_MISSING:
    df_train_mod = df_train_mod.dropna()
    df_test_mod = df_test_mod.dropna()
else:
    df_train_mod, df_test_mod = fill_missing_categorical(df_train_mod, df_test_mod, LABEL, missing_cat, 5)
    df_train_mod, df_test_mod = fill_missing_numerical(df_train_mod, df_test_mod, LABEL, missing_num, 5)
label_encoder = LabelEncoder()
df_train_mod[LABEL] = label_encoder.fit_transform(df_train_mod[LABEL])
# 保存到csv文件
df_train_mod.to_csv(f"train_mod_v{DATA_VERSION}.csv")
df_test_mod.to_csv(f"test_mod_v{DATA_VERSION}.csv")
encoders = {
    # 顺序编码
    'Drug': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=[['Placebo', 'D-penicillamine']]),
    # 顺序编码
    'Sex': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    # 顺序编码
    'Ascites': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    # 顺序编码
    'Hepatomegaly': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    # 顺序编码
    'Spiders': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1),
    # 独热编码
    'Edema': OneHotEncoder(),
    # 顺序编码
    'Stage': OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
}
for feat, enc in encoders.items():
    # 假如是顺序编码
    if isinstance(enc, OrdinalEncoder):
        # 在train数据集上训练OrdinalEncoder，并转换
        df_train_mod[feat] = enc.fit_transform(df_train_mod[[feat]]).astype('int32')
        # 根据训练好的OrdinalEncoder，预测test数据集
        df_test_mod[feat] = enc.transform(df_test_mod[[feat]]).astype('int32')
    
    # one-hot编码
    if isinstance(enc, OneHotEncoder):
        new_cols = enc.fit_transform(df_train_mod[[feat]]).toarray().astype('int8')
        col_names = enc.get_feature_names_out()

        df_train_mod[col_names] = new_cols
        df_train_mod.drop(feat, axis=1, inplace=True)
        
        # 对测试集进行编码
        new_cols_test = enc.transform(df_test_mod[[feat]]).toarray().astype('int8')
        df_test_mod[col_names] = new_cols_test
        df_test_mod.drop(feat, axis=1, inplace=True)
df_train_mod[NUM_FEATS].mean()
# 对填补好的train数据集深拷贝一份
tmp_df = df_train_mod.copy()

# 对数值型变量求均值
means = tmp_df[NUM_FEATS].mean()
std_devs = tmp_df[NUM_FEATS].std()

n_stds = 6
# 6sigma法则
thresholds = n_stds * std_devs

outliers = (np.abs(tmp_df[NUM_FEATS] - means) > thresholds).any(axis=1)

print(f"Detected {sum(outliers)} that are more than {n_stds} SDs away from mean...")
# The resulting boolean series can be used to filter out the outliers
outliers_df = tmp_df[outliers]

# Overwrite the train data
df_train_mod = tmp_df[~outliers].reset_index(drop=True)
print(f"Train data shape after outlier removal: {df_train_mod.shape}")
from typing import List, Dict

def validate_models(models: List[Dict],
                    data: pd.DataFrame, 
                    label=LABEL,
                    n_splits=5,
                    n_repeats=1,
                    seed=43):

    train_scores, val_scores = {}, {}
    
    pbar = tqdm(models)
    for model in pbar:
        
        model_str = model["name"]
        model_est = model["model"]
        model_feats = model["feats"]
        
        pbar.set_description(f"Processing {model_str}...")
        
        train_scores[model_str] = []
        val_scores[model_str] = []
        
        # 交叉检验
        skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

        for i, (train_idx, val_idx) in enumerate(skf.split(data[model_feats], data[label])):
            pbar.set_postfix_str(f"Fold {i+1}/{n_splits}")
            
            X_train, y_train = data[model_feats].loc[train_idx], data[label].loc[train_idx]
            X_val, y_val = data[model_feats].loc[val_idx], data[label].loc[val_idx]
            
            if model_str in ["lgb_cl"]:
                callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
                model_est.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            elif model_str in ["xgb_cl", "cat_cl"]:
                model_est.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)
            else:
                model_est.fit(X_train, y_train)
                
            train_preds = model_est.predict_proba(X_train[model_feats])
            valid_preds = model_est.predict_proba(X_val[model_feats])
            train_score = log_loss(y_train, train_preds)
            val_score = log_loss(y_val, valid_preds)
            train_scores[model_str].append(train_score)
            val_scores[model_str].append(val_score)
        
        model["avg_val_score"] = np.mean(val_scores[model_str])
            
    return models, pd.DataFrame(train_scores), pd.DataFrame(val_scores)
xgb_params = {
    'objective': 'multi_logloss', 
    'early_stopping_rounds': 50, 
    'max_depth': 9, 
    'min_child_weight': 8, 
    'learning_rate': 0.0337716365315986, 
    'n_estimators': 733, 
    'subsample': 0.6927955384688348, 
    'colsample_bytree': 0.1234702658812108, 
    'reg_alpha': 0.18561628377665318, 
    'reg_lambda': 0.5565488299127089, 
    'random_state': 42
}

rf_params = {'n_estimators':200}

xgb_cl = xgb.XGBClassifier(**xgb_params)
rf_cl = RandomForestClassifier(**rf_params)

FEATS = [c for c in df_train_mod.columns if c != LABEL]

models = [
    {"name": "xgb_cl", "model": xgb_cl, "feats": FEATS},
    {'name': "rf_cl", "model": rf_cl, "feats": FEATS}
]

models, train_scores, val_scores = validate_models(models=models, 
                                                   data=df_train_mod, 
                                                   n_splits=10,
                                                   n_repeats=1)
len(FEATS)
temp_array = np.random.rand(100, 20)
models[0]['model'].predict_proba(temp_array).shape
train_scores
val_scores