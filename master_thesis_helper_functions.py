import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage.filters import gaussian_filter
import numpy as np

def get_subset(df, constraints):
    """ 
    Function to get a subset of a dataframe based on certain values in a a given column
    input: df: Dataframe
           constraints: list of tuples containing column name and list of values of interest 
    return: subset Dataframe
    
    """
    for constraint in constraints:
        subset = df.loc[df[constraint[0]].isin(constraint[1])]
        df = subset
    return subset


def get_same_subset(df, subset):
    """ 
    Function to get a subset of a dataframe based on certain values in a a given column
    input: df: Dataframe
           constraints: list of tuples containing column name and list of values of interest 
    return: subset Dataframe
    
    """
    indices = [number -2  for number in subset.index]
    subset_new = df.iloc[indices, :]
    return subset_new


def get_distribution_from_question(question, df):
    """ 
    Function to get the number of occurences for each entry
    input: question: String, name of column for that question
    output: name: List of Strings, name of option
            number: List of Integers, number of occurneces for option in name at same index 
    
    """
    distribution = df[question].value_counts()
    name = []
    number = []
    for item in distribution.keys():
        name.append(item)
        number.append(distribution[item])  
    return name, number


def doghnut_plot(name, number):
    """ 
    Function to plot a doghnut distribution with the % of each option
    input: name: List of Strings, name of option
           number: List of Integers, number of occurneces for option in name at same index 
    
    """
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    # Give color names
    plt.pie(number, labels=name, colors=['red','green','blue','skyblue','purple', 'pink'])
    p=plt.gcf()
    p.gca().add_artist(my_circle)


def plot_country_distribution(country_dic):
    """
    from https://stackoverflow.com/questions/59297227/color-map-based-on-countries-frequency-counts
    """
    data = pd.DataFrame(country_dic).T.reset_index()
    data.columns=['country', 'count']
    
    # getting all the countries
    gapminder = px.data.gapminder().query("year==2007")

    countries=pd.merge(gapminder, data, how='left', on='country')

    fig = px.choropleth(countries, locations="iso_alpha",
                        color="count", 
                        hover_name="country", # column to add to hover information
                        color_continuous_scale=px.colors.sequential.Plasma)
    fig.show()
    return(gapminder["country"])


def heatmap(x, y, s, bins=1000):
    """
    from https://www.semicolonworld.com/question/43482/generate-a-heatmap-in-matplotlib-using-a-scatter-data-set
    """
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent


def addjitter(x,y,amp):
    x_new = []
    y_new = []
    for i in range(0,len(x)):
        rand_x = np.random.rand(1)[0]
        rand_y = np.random.rand(1)[0]
        jitter_x = (rand_x-0.5) * amp
        jitter_y = (rand_y-0.5) * amp
        x_new.append(x[i] + jitter_x)
        y_new.append(y[i] + jitter_y)
    return x_new, y_new
            