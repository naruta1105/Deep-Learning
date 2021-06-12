#### Exploratory Data Analysis (EDA) là một phương pháp phân tích dữ liệu 
#### chủ yếu sử dụng các kỹ thuật về biểu đồ, hình vẽ.

#### Importing Libraries ####
import pandas as pd
from dateutil import parser
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

#### Importing datasets ####
# from os.path import dirname
# from os import listdir
# Use os.path.basename(), os.path.dirname() to get the file name 
# and the directory name of the running file.
# print(listdir()) print all file in direction
# print(listdir()) 
# path_current_folder = dirname()
# path_data_file = path_current_folder +'/datasets/appdata10.csv'
dataset = pd.read_csv('datasets/appdata10.csv')


### EDA ###
# set show all column
pd.set_option('display.expand_frame_repr', False)
dataset.head() 
dataset.describe() # print out count for number of sample, mean

## Data Cleaning ##
dataset['hour'] = dataset.hour.str.slice(1,3).astype(int)

## Plotting ##
dataset2 = dataset.copy().drop(columns = ['user', 'screen_list', 'enrolled_date', 'first_open', 'enrolled'])
dataset2.head()

# Histograms #
# Name of histogram
plt.suptitle('Histograms of Numerical Columns', fontsize = 20)
# Create 7 sub histograms for 7 column from 1 -> 7
for i in range(1, dataset2.shape[1]+1):
    # size of subplot is 3x3
    plt.subplot(3, 3, i)
    # gca to cleans up everything that create field
    f = plt.gca()
    # make title for each sub plot
    f.set_title(dataset2.columns.values[i-1])
    
    # how many pins for each unique values of columns
    vals = np.size(dataset2.iloc[:, i-1].unique())
    
    # create histogramp with X = unique value, Y = frequency of this value
    plt.hist(dataset2.iloc[:, i-1], bins = vals, color = '#3F5D7D')
# Correlation with Response #
# xem tương quan giữa các thông số với cột Enrolled
# rot = 45  là gốc số liệu miêu tả của trục X là 45 độ
dataset2.corrwith(dataset.enrolled).plot.bar(figsize = (20,10),
                                             title = 'Correlation with Response Variable',
                                             fontsize = 15, rot = 45,
                                             grid = True)

# Correlation Matrix #
# fill the background
sn.set(style='white', font_scale=2)

# Compute the conrrelation matrix
# measure array of all the correlation of each field the choke go
corr = dataset2.corr()

# Generate a mask for the upper triangle
# create a mask for upper tray
# this map will display only a hafl of Square
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
# creat a suplot 18x15
f, ax = plt.subplots(figsize=(18, 15))
f.suptitle('Correlation Matrix', fontsize = 40)

# Generate a custom diverging colormap
# make a colormap (màu của ô vuông)
cmap =  sn.diverging_palette(220, 10, as_cmap = True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(corr, mask = mask, cmap = cmap, vmax = .3, center = 0,
           square = True, linewidths = 5, cbar_kws = {"shrink": .5})

### Feature Engineering ###
# Check type of all feature
dataset.dtypes

dataset["first_open"] = [parser.parse(row_data) for row_data in dataset["first_open"]]
dataset["enrolled_date"] = [parser.parse(row_data) if isinstance(row_data, str) else row_data for row_data in dataset["enrolled_date"]]
dataset.dtypes
##astype(timedelta64[h]) to convert to Hours
dataset['difference'] = (dataset.enrolled_date - dataset.first_open).astype('timedelta64[h]')

plt.hist(dataset['difference'].dropna(), color = '#3F5D7D')
plt.title('Distribution of Time-Since-Enrolled')
plt.show()
# => we see that in people enroll in first 100 hour but we not know excatly => create diff histogram

plt.hist(dataset['difference'].dropna(), color = '#3F5D7D', range = [0,100])
plt.title('Distribution of Time-Since-Enrolled')
plt.show()
# => we see that in people enroll in first 10 hour, and there is in 25 hour => choose 48 hours

# change every row in 'enrolled' column which difference more than 48 to 0
dataset.loc[dataset.difference > 48, 'enrolled'] = 0
# remove the difference,'enrolled_date', 'first_open' colums
dataset = dataset.drop(columns = ['difference', 'enrolled_date', 'first_open'])

## Formattingthe screen_list field

top_screens = pd.read_csv('datasets/top_screens.csv').top_screens.values

dataset["screen_list"] = dataset.screen_list.astype(str) + ','

for sc in top_screens:
    dataset[sc] = dataset.screen_list.str.contains(sc).astype(int)
    dataset["screen_list"] = dataset.screen_list.str.replace(sc+",","")

# count other screen not in top_screen file
dataset["Other"] = dataset.screen_list.str.count(",")

# remove screen_list column
dataset = dataset.drop(columns = ['screen_list'])

# Funnels
# Funnels là những screen đi chung 1 nhóm. Ta nhóm những screen đó vào 1 nhóm
# Thêm 1 cột là tên nhóm và bỏ đi những cột tên screen đó
savings_screens = ["Saving1",
                    "Saving2",
                    "Saving2Amount",
                    "Saving4",
                    "Saving5",
                    "Saving6",
                    "Saving7",
                    "Saving8",
                    "Saving9",
                    "Saving10"]
dataset["SavingCount"] = dataset[savings_screens].sum(axis=1)
dataset = dataset.drop(columns=savings_screens)

cm_screens = ["Credit1",
            "Credit2",
            "Credit3",
            "Credit3Container",
            "Credit3Dashboard"]
dataset["CMCount"] = dataset[cm_screens].sum(axis=1)
dataset = dataset.drop(columns=cm_screens)

cc_screens = ["CC1",
                "CC1Category",
                "CC3"]
dataset["CCCount"] = dataset[cc_screens].sum(axis=1)
dataset = dataset.drop(columns=cc_screens)

loan_screens = ["Loan",
            "Loan2",
            "Loan3",
            "Loan4"]
dataset["LoansCount"] = dataset[loan_screens].sum(axis=1)
dataset = dataset.drop(columns=loan_screens)

#### Saving Results ####
dataset.head()
dataset.describe()
dataset.columns

dataset.to_csv('new_appdata10.csv', index = False)
