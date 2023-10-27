import os
import pandas as pd

import numpy as np

#insert path to your folder Figure 15

us_folder='users'
users_info = []
ids=[]
for fold in os.listdir(us_folder):
    if fold.split('_')[0] == 'user':
        id=fold.split('_')[1]
        print(id)
        user_info=[]
        with open(us_folder+'/'+fold+'\\'+ 'save_personal_info.txt', 'r') as fp:
            #read each line of file
            for line in fp:
                l=line.replace('\n', '')
                if len(l.split('_'))>1:
                    if l.split('_')[1]=='':
                        user_info.append('None')
                    else:
                        user_info.append(l.split('_')[1])
                else:
                    user_info.append('None')
                print(l)
        users_info.append([i for i in user_info])
        ids.append(id)
        print('-----------------')

users_info_df=pd.DataFrame(users_info)
#put rows to eliminate

#remove columns in users_info_df from 9 to the end
users_info_df=users_info_df.drop(users_info_df.columns[9:], axis=1)
#add a new column called resolution with the values of column1 + column2
users_info_df['resolution']=users_info_df[1]+'x'+users_info_df[2]
#put ids as index
users_info_df.index=ids
#remove rows 49f600b7-c387-4e68-bc65-36b66f64a8bf 9fd7025d-5ebf-40d8-9d4b-6487f00ae9d1
users_info_df=users_info_df.drop(['49f600b7-c387-4e68-bc65-36b66f64a8bf','9fd7025d-5ebf-40d8-9d4b-6487f00ae9d1'], axis=0)

#do the same for these: 32067447-5118-40ff-b0bf-d85b0dadd388 c86c37ea-3e1d-41ee-b860-88fd71e68ee3 28160ab7-5ea9-40a3-9ffc-41be3098ec7e
users_info_df=users_info_df.drop(['32067447-5118-40ff-b0bf-d85b0dadd388','c86c37ea-3e1d-41ee-b860-88fd71e68ee3','28160ab7-5ea9-40a3-9ffc-41be3098ec7e'], axis=0)





#save users_info_df to excel
users_info_df.to_excel('users_info.xlsx', index=True)

#calculate value for each column
for c in users_info_df.columns:
    print("---- %s ---" % c)
    print(users_info_df[c].value_counts()*100/len(users_info_df))