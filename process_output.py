import pandas as pd


df = pd.read_csv('./output.csv')
test_dataset = pd.read_csv('./datasets/test_dataset.csv')


all_test = list(set(list(test_dataset['user_id'])))

df.drop(df.columns[[0]],axis=1,inplace=True)
#print(df.head())
#print(df.index)

df =df.groupby(['userID'])[['itemID','prediction']].apply(lambda x:x.sort_values(by=['prediction'],ascending=False)[:10])

print(type(df))
print(df.head())

idx = 0
print(len(all_test))
df2 = pd.DataFrame(columns=['userID','itemID'])
dfoutput = pd.DataFrame(columns=['userID','itemID'])
dfl = []
for d in all_test:
    print(d)
    dfx = df.loc[d].copy(deep=True).reset_index(drop=True)
    #print(dfx)
    dfx.drop(dfx.columns[[1]],axis=1,inplace=True)
    dfx.insert(0, 'userID', d)
    dfl.append(dfx)

dfoutput = pd.concat(dfl,axis=0)


dfoutput.to_csv('submission_2.csv',index=False)
