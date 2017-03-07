##2 Install Pandas
#pip install pandas



##11 Pandas basic
#like a dictionary
import pandas as pd
import numpy as np
s = pd.Series([1,2,3,4, np.nan,52,1])
s

#6 dates from today
dates = pd.date_range('20170227',periods=6)
dates

#create 6x4 matrix with row name from dates, col name abcd
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
df

#create 3x4 matrix with default row-colname
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
df1

#create by Dic assigning each column
df2 = pd.DataFrame({'A':1.,
    'B':pd.Timestamp('20170227'),
    'C':pd.Series(1,index=list(range(4)),dtype='float32'), 
    'D':np.array([3]*4,dtype='int32'),
    'E':pd.Categorical(['test','train','test','train']),
    'F':'foo'})
df2

df2.dtypes #show type for each column
df2.columns #show colname 
df2.values #show each value within
df2.describe() #show summary for each column (only with metric)
df2.T #transpose dataframe

#sorting according to colname(axis = 1) or rowname(axis=0)
df2.sort_index(axis=1,ascending=False)

#sorting according to value (by column 'E')
df2.sort_values(by='E')



#12 Pandas select
dates = pd.date_range('20170101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
df

#choose one column 'A'
print(df['A'],df.A)
#choose certain row
print(df[0:3],df['20170102':'20170104'])

#m1:select by label: loc (locate with certain index label, could assign label value)
print(df.loc['20170102'])
#select by location and colname
print(df.loc[:,['A','B']])
#more specific row, col
print(df.loc['20170102',['A','B']])

#m2:select by position: iloc (locate with certain index)
print(df.iloc[3])
print(df.iloc[3,1])
print(df.iloc[3,1:3])
print(df.iloc[3:5,1:3])
print(df.iloc[[1,3,5],1:3])

#m3:select by mix: ix (locate with index&index label)
print(df.ix[:3,['A','C']])

#m4:Boolean indexing
print(df)
print(df[df.A>8]) #only show matrix with column'A' value>8
print(df[df['A']<8])



##13 Pandas assign new value
dates = pd.date_range('20170101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
df

df.iloc[2,2] = 1111 #assign new value to (2,2)
df.loc['20170101','B'] = 2222 #assign new value to (20170101,B) label pair
df.ix[3,'A'] = 3142
df[df.A>4]=0 #change element in matrix=0 if col'A' value > 4
df.A[df.A>4]=0 #change element in col'A' if col'A' value > 4
df.B[df.A>4]=0 #change element in col'A' if col'A' value > 4

#add extra col'F' with NAN
df['F'] = np.nan
#add extra col'E' with assign series
df['G'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20170101',periods=6))
#note the date_range (or list) should align, or will put to corresponding position directly
print(df)



##14 Pandas missing dealing
dates = pd.date_range('20170101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates,columns=['A','B','C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan #missing part
df

#m1: by row(axis=0) defaule 'any'
print(df.dropna(axis=0,how='any')) #any: 1nan will drop, all: allnan will drop
#m2: by col(axis=1)
print(df.dropna(axis=1,how='all')) #keep all the same

#impute
print(df.fillna(value=0)) #NaN fill with 0

#detect
print(df.isnull()) #show matrix where NaN
print(np.any(df.isnull())==True) #if matrix too big, only show True when Nan presented



##15 Pandas input output
#read_csv, read_json, read_pickle...
#to_csv, to_excel...
#load
data = pd.read_csv('C:\\Users\\jncinlee\\Desktop\\gitTUT\\PastHires.csv') #para sep....
print(data)

#save
data.to_pickle('C:\\Users\\jncinlee\\Desktop\\gitTUT\\PH.pickle')



##16 Pandas concat
###concatenate
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
print(df1,df2,df3)

#up-low combine(axis=0)
res = pd.concat([df1,df2,df3],axis=0)
print(res)

res = pd.concat([df1,df2,df3],axis=0,ignore_index=True) #combine with new index
print(res)

###join inner, outer
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
print(df1,df2)

res=pd.concat([df1,df2]) #hard combine fill with NaN, default as outer
print(res)
#inner only consider both have in column ie.(b,c,d) and up-low combine
res=pd.concat([df1,df2],join='inner',ignore_index=True) 
print(res)


#left-right combine(axis=1)
###join axis
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
res = pd.concat([df1,df2],axis=1,join_axes=[df1.index]) 
#left-right combine but impute NaN for df2 and cut below where df1.index don't have
print(res)

res = pd.concat([df1,df2],axis=1) #consider both indexes
print(res)


#panda append, add any new examples below
###append
df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])

#res = df1.append(df2,ignore_index=True) #add df2 below
res = df1.append([df2,df3],ignore_index=True) #add df2,df3 below, no assign will be Nan
print res

#append a series
s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res = df1.append(s1,ignore_index=True) #ignore_index must there, one at a time
print res



#17 Pandas merge
left=pd.DataFrame({'key':['k0','k1','k2','k3'],
                   'A':['a0','a1','a2','a3'],
                   'B':['b0','b1','b2','b3']})
right=pd.DataFrame({'key':['k0','k1','k2','k3'],
                   'C':['c0','c1','c2','c3'],
                   'D':['d0','d1','d2','d3']})
print(left,right)
#merge by key
res = pd.merge(left,right,on='key')
print res

#consider two key
left=pd.DataFrame({'key1':['k0','k0','k1','k2'],
                   'key2':['k0','k1','k0','k1'],
                   'A':['a0','a1','a2','a3'],
                   'B':['b0','b1','b2','b3']})
right=pd.DataFrame({'key1':['k0','k1','k1','k2'],
                   'key2':['k0','k0','k0','k0'],
                   'C':['c0','c1','c2','c3'],
                   'D':['d0','d1','d2','d3']})
print left, right
#inner
res = pd.merge(left,right,on=['key1','key2'],how='inner') 
#inner only consider complete match key1,key2 
#evenif 2 same key k1,k0 left example replicate 2 times
print res

#outer
res = pd.merge(left,right,on=['key1','key2'],how='outer') 
#complete keep all key
print res

#right
res = pd.merge(left,right,on=['key1','key2'],how='right') 
#base on right key, unknown left will be NaN
print res

#right
res = pd.merge(left,right,on=['key1','key2'],how='left') 
#base on left key, unknown right will be NaN
print res


#indicator
df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print df1,df2
res = pd.merge(df1,df2,on='col1',how='outer',indicator=True)
#will show how it's merged
res = pd.merge(df1,df2,on='col1',how='outer',indicator='indicolumn') #call as indicolumn
print res


#merge by index, both given index
left=pd.DataFrame({'A':['a0','a1','a2','a3'],
                   'B':['b0','b1','b2','b3']},
                   index=['k0','k1','k2','k4'])
right=pd.DataFrame({'C':['c0','c1','c2','c3'],
                   'D':['d0','d1','d2','d3']},
                   index=['k2','k3','k4','k5'])
print(left,right)
#turn on the index value as key
res = pd.merge(left,right,left_index=True,right_index=True,how='outer')
print res


#dealing overlap colname
boys = pd.DataFrame({'k':['k0','k1','k2'],'age':[1,2,3]})
girls = pd.DataFrame({'k':['k0','k0','k3'],'age':[4,5,6]})
res = pd.merge(boys,girls,on='k',how='inner',suffixes=['_boy','_girl'])
#add suffix if for same colname
print boys; print girls; print res



##18 Pandas plot
import matplotlib.pyplot as plt
#plot data

#series
data = pd.Series(np.random.randn(1000),index = np.arange(1000))
data = data.cumsum()
data.plot()
plt.show()

#dataframe
data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list("abcd"))
data = data.cumsum()
#print data.head()
data.plot()
plt.show() #plot 4 set of it based on a,b,c,d

#.bar .hist .box .kde .area .scatter .hexbin .pie
ax = data.plot.scatter(x='a',y='b',color='darkblue',label='class1')
data.plot.scatter(x='a',y='c',color='darkgreen',label='class2', ax = ax) #remember giving ax
plt.show()#
