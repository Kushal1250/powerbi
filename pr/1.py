# Write a Python script to plot a line graph that shows the yearly revenue of a
# company (from 2015 to 2024). Ensure the following:
# -The x-axis represents years, and the y-axis represents revenue in million dollars.
# -The title should be "Company Revenue Over Years".
# -Use a purple color dashed line ('r--') for plotting.

# import matplotlib.pyplot as plt
# years = list(range(2015, 2026))
# revenue = [50, 60, 68, 75, 80, 85, 90, 95, 110, 120,1]
# plt.plot(years, revenue, 'm--', marker='o')  
# plt.title("Company Revenue Over Years")
# plt.xlabel("Year")
# plt.ylabel("Revenue (in Million Dollars)")
# plt.grid(True)
# plt.legend(['Revenue'])
# plt.show()


# What is the difference between a bar chart and a histogram? Explain with
# examples.
# categorical datatype
# import matplotlib.pyplot as plt

# products = ['A', 'B', 'C', 'D']
# sales = [150, 200, 120, 180]

# plt.bar(products, sales, color='skyblue', edgecolor='black')
# plt.title("Product Sales")
# plt.xlabel("Products")
# plt.ylabel("Sales (Units)")
# plt.show()

# # continuous data type
# import matplotlib.pyplot as plt

# scores = [45, 48, 52, 55, 60, 61, 65, 66, 70, 72, 74, 75, 80, 84, 85, 88, 90, 92, 95, 98]

# plt.hist(scores, bins=15, color='orange' , edgecolor='black')
# plt.title("Distribution of Test Scores")
# plt.xlabel("Score Ranges")
# plt.ylabel("Number of Students")
# plt.show()
# plt.hist()

# Write a Python script to plot a histogram of students' marks in a class. Ensure:
# -Use 50 random marks between 0 to 100.
# -Set bins = 10.
# -X-axis: Marks range, Y-axis: Number of students.
# -Add a grid for better readability.


# Write a Python script to plot a histogram of students' marks in a class. Ensure:
# -Use 50 random marks between 0 to 100.
# -Set bins = 10.
# -X-axis: Marks range, Y-axis: Number of students.
# -Add a grid for better readability.

# import matplotlib.pyplot as plt
# import numpy as np
# marks = np.random.randint(0, 101, size=50)
# plt.hist(marks, bins=10, color='skyblue', edgecolor='black')
# plt.title("Distribution of Students' Marks")
# plt.xlabel("Marks Range")
# plt.ylabel("Number of Students")
# plt.grid(True)
# plt.show()



# How do you create a scatter plot in Matplotlib? Write a 
# Python script to visualize
# the relationship between advertising budget 
# and sales revenue of a company.
# Use: X-axis: Advertising Budget ($), Y-axis: Sales Revenue ($).
# Use blue circular markers ('bo').
# Include a regression line if applicable.

# import matplotlib.pyplot as plt
# import numpy as np
# ad_budget = np.array([500, 700, 1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500])
# sales_revenue = np.array([5500, 5800, 6000, 6200, 6800, 7000, 7300, 7800, 8200, 9000])
# plt.scatter(ad_budget, sales_revenue, color='blue', marker='o', label='Data Points')
# coefficients = np.polyfit(ad_budget, sales_revenue, 1)  # degree=1 for linear
# regression_line = np.poly1d(coefficients)
# x_vals = np.linspace(min(ad_budget), max(ad_budget), 100)
# plt.plot(x_vals, regression_line(x_vals), color='red', label='Regression Line')
# plt.title("Advertising Budget vs Sales Revenue")
# plt.xlabel("Advertising Budget ($)")
# plt.ylabel("Sales Revenue ($)")
# plt.grid(True)
# plt.legend()
# plt.show()
# plt.plot()

# Write a Python program to create a pie chart showing the market share of
# different smartphone brands. Ensure:
# The title should be "Smartphone Market Share 2024".
# Use percentage labels and an exploded slice for the leading brand.
# Use different colors for each segment.

# import matplotlib.pyplot as plt
# brands = ['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E']
# market_share = [35, 25, 20, 10, 10]  # in percentage
# explode = [0.1, 0, 0, 0, 0]
# colors = ['gold', 'skyblue', 'lightcoral', 'lightgreen', 'violet']
# plt.figure(figsize=(8, 8))
# plt.pie(market_share,labels=brands,explode=explode,colors=colors,autopct='%1.1f%%',shadow=True,startangle=140)
# plt.title("Smartphone Market Share 2024")
# plt.axis('equal')
# plt.show()

# import matpl
# 
# 
# 
# 
# 
# 
# monthly_temps = [
#     [7, 6, 8, 9, 5],   # Jan
#     [9, 10, 11, 8, 12],# Feb
#     [13, 15, 14, 12, 16], # Mar
#     [17, 19, 18, 16, 20], # Apr
#     [22, 23, 21, 24, 25], # May
#     [28, 27, 29, 30, 26], # Jun
#     [32, 33, 31, 30, 34], # Jul
#     [31, 30, 32, 33, 29], # Aug
#     [27, 26, 25, 28, 24], # Sep
#     [20, 21, 19, 22, 23], # Oct
#     [14, 13, 12, 15, 11], # Nov
#     [9, 8, 10, 7, 11]     # Dec
# ]

# months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
#           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# plt.figure(figsize=(12, 6))
# plt.boxplot(monthly_temps,patch_artist=True, boxprops=dict(facecolor='lightblue'))
# plt.title("Monthly Temperatures Box Plot (City XYZ)")
# plt.xlabel("Month")
# plt.ylabel("Temperature (°C)")
# plt.xticks(ticks=range(1, 13), labels=months)
# plt.grid(True)
# plt.show()




# import matplotlib.pyplot as plt

# # Create a figure and axis
# fig, ax = plt.subplots()

# # Walls (rectangle)
# walls_x = [1, 1, 5, 5, 1]
# walls_y = [1, 4, 4, 1, 1]
# ax.plot(walls_x, walls_y, 'brown', label='Walls')

# # Roof (triangle)
# roof_x = [1, 3, 5]
# roof_y = [4, 6, 4]
# ax.plot(roof_x, roof_y, 'red', label='Roof')

# # Door (rectangle)
# door_x = [2, 2, 3, 3, 2]
# door_y = [1, 3, 3, 1, 1]
# ax.plot(door_x, door_y, 'blue', label='Door')

# # Window 1 (rectangle)
# window1_x = [3.5, 3.5, 4.5, 4.5, 3.5]
# window1_y = [3, 4, 4, 3, 3]
# ax.plot(window1_x, window1_y, 'cyan', label='Window 1')

# # Window 2 (rectangle)
# window2_x = [4, 4, 4.5, 4.5, 4]
# window2_y = [1.5, 2, 2, 1.5, 1.5]
# ax.plot(window2_x, window2_y, 'magenta', label='Window 2')
# # Title and labels
# ax.set_title("House Using Line Plots with Multiple Colors")
# ax.set_xlabel("X-axis")
# ax.set_ylabel("Y-axis")
# ax.legend()
# ax.grid(True)

# # Show plot
# plt.show()




# import seaborn as sns
# import matplotlib.pyplot as plt
# tips = sns.load_dataset("tips")
# sns.relplot(
#     data=tips,
#     x="day", y="tip",
#     kind="line",
#     hue="sex",              
#     style="smoker",         
#     dashes={"Yes": (2, 2), "No": (1, 1)},  
#     markers={"Yes": "o", "No": "s"},       
#     marker=True,
#      )
# plt.title("Average Tip by Day, Gender, and Smoking Status")
# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# sns.regplot(x="total_bill", y="tip", data=tips)
# plt.title("Regression Plot: Total Bill vs Tip")
# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the fmri dataset
# fmri = sns.load_dataset("fmri")

# # Create relational line plot with standard deviation error bars
# sns.relplot(
#     data=fmri,
#     x="timepoint",
#     y="signal",
#     kind="line",
#     errorbar="sd"
# )

# plt.title("FMRI Signal Over Time with Standard Deviation")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the tips dataset
# tips = sns.load_dataset("tips")

# # Create displot with custom bin edges
# sns.displot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])

# plt.title("Distribution of Table Sizes")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the tips dataset
# penguins = sns.load_dataset("penguins")

# # Create displot with custom bin edges
# sns.displot(data=penguins, x="flipper_length_mm",hue='species',multiple='stack')
# plt.title("Distribution of Table Sizes")
# plt.show()




# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# plt.figure(figsize=(12, 4))

# # No jitter
# plt.subplot(2, 3, 1)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=0)
# plt.title("jitter=0 (No jitter)")

# # Moderate jitter
# plt.subplot(2, 3, 2)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=0.2)
# plt.title("jitter=0.2")

# plt.subplot(2, 3, 3)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=0.4)
# plt.title("jitter=0.4")

# plt.subplot(2, 3, 4)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=5.5)
# plt.title("jitter=0.5")

# plt.tight_layout()
# plt.show()






# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# sns.stripplot(data=tips, x="smoker", y="tip", order=["No", "Yes"])

# plt.title("Tip Amount by Smoking Status (Ordered)")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# # Create violin plot without inner markings
# g = sns.catplot(data=tips, x="day", y="total_bill", kind="violin", inner=None)

# # Overlay swarm plot on the same axes
# sns.swarmplot(data=tips, x="day", y="total_bill", color="k", size=3, ax=g.ax)

# plt.title("Total Bill Distribution by Day with Individual Points")
# plt.show()

# import seaborn as sns
# import matplotlib.pyplot as plt
# fmri = sns.load_dataset("fmri")

# sns.relplot(
#     data=fmri,
#     x="timepoint", y="signal",
#     kind="line",
#     hue="event",
#     col="region"  # Facet by 'region' (creates one plot per region)
# )
# plt.show()

# tips = sns.load_dataset("tips")

# sns.displot(
#     data=tips,
#     x="total_bill",
#     col="time",  # Facet by 'time' (Lunch, Dinner)
#     kde=True
# )

# sns.catplot(
#     data=tips,
#     x="day", y="total_bill",
#     kind="box",
#     row="smoker"  # Facet rows by smoker status
# )
# tips = sns.load_dataset("tips")

# sns.lmplot(
#     data=tips,
#     x="total_bill", y="tip",
#     hue="smoker",
#     col="time"  # Facet by time of day (Lunch/Dinner)
# )
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# fmri = sns.load_dataset("fmri")

# # Create FacetGrid by 'event'
# g = sns.FacetGrid(fmri, col="event")

# # Map scatter plot on each facet
# g.map(sns.scatterplot, "timepoint", "signal")

# # Overlay line plot on each facet
# g.map(sns.lineplot, "timepoint", "signal")

# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")
# plt.figure(figsize=(8, 6))
# sns.boxplot(data=tips, x="day", y="tip")
# sns.swarmplot(data=tips, x="day", y="tip", color="k", size=3)
# plt.title("Tip Distribution by Day with Individual Points")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# plt.figure(figsize=(8, 6))

# # Split violin plot comparing smokers vs non-smokers on total_bill
# sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker", split=True)

# # Overlay swarm plot for individual data points
# sns.swarmplot(data=tips, x="day", y="total_bill", hue="smoker", dodge=True, color="k", size=3)

# plt.title("Total Bill Distribution by Day Split by Smoker Status")
# plt.legend(title="Smoker", loc='upper left')
# plt.grid(True)
# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# plt.figure(figsize=(8, 6))

# # Violin plot shows the distribution shape and density
# sns.violinplot(data=tips, x="day", y="total_bill", inner=None, color=".8")

# # Strip plot overlays individual data points with jitter for visibility
# sns.stripplot(data=tips, x="day", y="total_bill", jitter=True, color="b", size=4)

# plt.title("Total Bill Distribution by Day with Individual Data Points")
# plt.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# sns.catplot(
#     data=tips,
#     x="day",
#     y="total_bill",
#     kind="strip",
#     jitter=True,
#     hue="sex",
#     col="time",
#     palette="muted"
# )

# plt.show()


# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the fmri dataset
# fmri = sns.load_dataset("fmri")

# # Create the relational line plot
# sns.relplot(
#     data=fmri,
#     x="timepoint", y="signal",
#     kind="line",
#     hue="region",        # Line color by region (parietal, frontal)
#     style="event",       # Line style and markers by event (stim, cue)
#     markers=True,        # Enable markers
#     dashes=True,         # Enable dashed lines for styles
#     ci="sd"              # Show standard deviation as shaded area
# )

# # Set title and display the plot
# plt.title("FMRI Signal Over Time by Region and Event")
# plt.show()


# import pandas as pd
# import plotly as px

# df=pd.read_csv('1.csv')
# df['Date']=pd.to_datetime(df['Date'])
# df=df.sort_values('Date')

# fig=px.line(df,
#             x='Date',
#             y='Close_time',
#             color='Stock',
#             line_group='Stock',
#             animation_layout=df['Date'].dt.strftime('$Y-%m-%d'),
#             hover_update={'Stock':True,'Date':True,'Close_time':True},
#             labels={"Close_time":'price($)'},
#             title='Stock Price Over Time',)

# fig.update_layout(
#     updatemenu=[
#         {
#            "button":[
#               {
#                   'method': 'update',
#                   'label': stock,
#                   'args':[{"visible":[s==stock for s in df['date'].unique()]},
#                           {"title":f"stock:{stock}"}]
#               }for stock in df['Stock'].unique()
#               ],
#                 'direction': 'down',
#                 'showactive': True,
                   
#               } 
#            ],
#            xaxis_labe="date",
#            yaxis_label="price($)", 
#             hovemode='x',
# )
# fig.update_layout(
#      dragemode='pan',
#      xaxis=dict(rangeslider_visible=True), 
#      yaxis=dict(fixedrange=False)  
# )



# import plotly.graph_objects as go

# fig = go.Figure(data=go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[10, 15, 13, 17],
#     mode='markers',
#     marker=dict(
#         color='red',  # Static color
#         size=12
#     )
# ))
# fig.show()


# import plotly.graph_objects as go

# fig=go.Figure(data=go.Scatter(
#     x=[1,2,3,4],
#     y=[1,2,3,4],
#     mode='markers',
#     marker=dict(
#         color='red',
#         size=3
#     )
# ))
# fig.show()


# import plotly.express as px
# import pandas as pd

# # Sample data
# data = pd.DataFrame({
#     'Date': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04'],
#     'Sales': [150, 200, 170, 220]
# })

# # Create line chart
# fig = px.line(data, x='Date', y='Sales', title='Sales Over Time')
# fig.show()




# import plotly.graph_objects as go

# fig = go.Figure()

# fig.add_trace(go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[10, 15, 13, 17],
#     mode='lines',
#     name='Trace A'
# ))

# fig.add_trace(go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[16, 5, 11, 9],
#     mode='lines',
#     name='Trace B'
# ))

# fig.update_layout(title="Click Legend to Toggle Traces")
# fig.show()


# import plotly.express as px
# import pandas as pd

# # Sample DataFrame
# df = pd.DataFrame({
#     'Product': ['Product A', 'Product B', 'Product C'],
#     'Sales': [100, 150, 80]
# })

# # Create bar chart with axis labels
# fig = px.bar(df, x='Product', y='Sales', title='Sales by Product',
#              labels={'Product': 'Product Name', 'Sales': 'Sales Amount'})
# fig.show()
# import plotly.graph_objects as go
# label=['applee','banana','orange']
# value=[50,28,22]    
# customer_hover=['Fresh red apples', 'Sweet ripe bananas', 'Juicy cherries']
# fig=go.Figure(data=[go.Pie(
#     labels=label, 
#     values=value, 
#     hovertext=customer_hover,
#     hoverinfo='text+percentage+label'
# )])
# fig.update_layout(title_text='Fruit Sales Distribution')
# fig.show()



# import seaborn as sns
# import matplotlib.pyplot as plt

# tips = sns.load_dataset("tips")

# # sns.histplot(data=tips, x="total_bill", bins=20, kde=False)
# # plt.title("Histogram of Total Bill")
# # plt.show()
# sns.kdeplot(data=tips, x="total_bill", fill=True)
# plt.title("KDE of Total Bill")
# plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Load the fmri dataset
fmri = sns.load_dataset("fmri")

# Create the relational line plot
sns.relplot(
    data=fmri,
    x="timepoint", y="signal",
    kind="line",
    hue="event",        # Line color by region (parietal, frontal)
    style="region",       # Line style and markers by event (stim, cue)
    markers=True,        # Enable markers
    dashes=True,         # Enable dashed lines for styles
    ci="sd"              # Show standard deviation as shaded area
)

# Set title and display the plot
plt.title("FMRI Signal Over Time by Region and Event")
plt.show()
