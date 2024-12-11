import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_theme(style = 'dark')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import scipy.stats as stats
from sklearn.linear_model import LinearRegression


# Set page configuration
st.set_page_config(layout = "wide")

# Function to load sample datasets
def load_dataset(name):
    datasets = {
        "Mtcars": sns.load_dataset("mpg").dropna() if "mpg" in sns.get_dataset_names() else None,
        "Iris": sns.load_dataset("iris"),
        "Titanic": sns.load_dataset("titanic").dropna() if "titanic" in sns.get_dataset_names() else None,
        "Tips": sns.load_dataset("tips").dropna() if "tips" in sns.get_dataset_names() else None,        
        "Penguins": sns.load_dataset("penguins").dropna() if "penguins" in sns.get_dataset_names() else None,
        "Glue": sns.load_dataset("glue").dropna() if "glue" in sns.get_dataset_names() else None,
        "Car-crashes": sns.load_dataset("car_crashes").dropna() if "car_crashes" in sns.get_dataset_names() else None,
        "Planets": sns.load_dataset("planets").dropna() if "planets" in sns.get_dataset_names() else None
    }
    return datasets.get(name, None)

# Define dataset descriptions
dataset_descriptions = {
    "Mtcars": "[Mtcars](https://www.geeksforgeeks.org/seaborn-datasets-for-data-science/#8-mpg-dataset)",
    "Iris": "[Iris](https://en.wikipedia.org/wiki/Iris_flower_data_set)",
    "Titanic": "[Titanic](https://www.kaggle.com/c/titanic/data)",
    "Tips": "[Tips](https://rdrr.io/cran/reshape2/man/tips.html)",
    "Penguins": "[Penguins](https://allisonhorst.github.io/palmerpenguins/reference/penguins_raw.html)",
    "Glue": "[Glue](https://www.kaggle.com/code/ellekayem/data-visualization-with-the-glue-dataset#First-Conclusions)",
    "Car-crashes": "[Car-crashes](https://www.kaggle.com/fivethirtyeight/fivethirtyeight-bad-drivers-dataset)",
    "Planets": "[Planets](https://www.geeksforgeeks.org/seaborn-datasets-for-data-science/#9-planets-dataset)",
}

# Sidebar for dataset selection
st.sidebar.title("An introduction to Data Visualization")
dataset_choice = st.sidebar.selectbox(
    "Select Dataset",
    ["Upload", 'Car-crashes', 'Glue', "Iris", "Mtcars", 'Penguins', 'Planets', 'Tips', "Titanic"]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV File", type = ["csv"])

# Data Loading
try:
    if dataset_choice == "Upload" and uploaded_file:
        delim = st.sidebar.text_input("Delimiter", "")
        data = pd.read_csv(uploaded_file, delimiter = delim if delim else None, low_memory = False)
    
    else:
        data = load_dataset(dataset_choice)
        if data is None:
            st.warning(f"Please select a dataset or upload a csv file.")
except Exception as e:
    st.error(f"Error loading the dataset: {e}")
    data = None

# Data preview and validation

numeric_var = []
if data is not None:
    st.sidebar.write(f"Dataset shape: {data.shape}")
    num_rows = st.sidebar.slider("Number of rows to display", 5, 50, 10)
    for i in data.columns:
        if data[i].dtype in [float, int]:
            data[i] = pd.to_numeric(data[i], errors = 'coerce')
            numeric_var.append(i) 

print("Numeric transform finished")
# Sidebar for grouping numeric variables
num_var = st.sidebar.selectbox(
    "Select Numeric Variable to Group",
    ["None"] + numeric_var 
)
print("At Line 68")
group_thresholds = st.sidebar.text_input("Enter Thresholds (comma-separated)", "")
group_names = []
if num_var != "None" and group_thresholds:
    try:
        thresholds = [float(x.strip()) for x in group_thresholds.split(",")]
        for i in range(len(thresholds) + 1):
            default_name = f"Category {i+1}"
            group_name = st.sidebar.text_input(f"Name for Category {i+1}", value = default_name)
            group_names.append(group_name)

        if st.sidebar.button("Apply Grouping"):
            if len(group_names) == len(thresholds) + 1:
                bins = [-np.inf] + thresholds + [np.inf]
#                labels = group_names
                data[f"{num_var}_grouped"] = pd.cut(data[num_var], bins = bins, labels = group_names)
                # Store modified data in session state
                st.session_state.data = data
            
            else:
                st.error("Number of thresholds and group names must match.")
    except ValueError:
        st.error("Thresholds must be numeric values.")

# Use session state if the data exists
if 'data' in st.session_state:
    data = st.session_state.data

if data is not None:
    cat_var = []
    for i in data.columns:
        if data[i].dtype not in [float, int]:
            data[i] = data[i].apply(lambda x: str(x) if x != np.nan else x)
            cat_var.append(i) 

st.sidebar.markdown("---")
st.sidebar.markdown(
    "🔗 **Check out the [Codes](https://github.com/SaifurRahmanShishir/Visuals_in_-Streamlit)**",
    unsafe_allow_html=True
)
# Tabs for visualization and data insights
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview & Summary", "Univariate Plots", "Bivariate Plots", "Multivariate Plot"])

# Tab 1: Data preview
with tab1:
    if data is not None:
        st.write('### Preview')
        st.write(data.head(n = num_rows))
        st.write("### Data Summary")
        st.write(data.describe(include="all"))

        if dataset_choice in dataset_descriptions:
            st.markdown(
            f"**Dataset Description:** {dataset_descriptions[dataset_choice]}",
            unsafe_allow_html = True
        )
        else:
            st.markdown(
            "**Dataset Description:** No specific description available for the uploaded dataset."
        )



# Tab 2: Univariate Plots
with tab2:
    
    if data is not None:
        st.write("### Univariate Analysis")
        col = st.selectbox("Select Column for Univariate Analysis", data.columns)
    
        # Filter `cat_var` to exclude the selected column `col`
        filtered_cat_var = [c for c in cat_var if c != col]
    
        # Select Group By Column (zvar) with the filtered list
        zvar = st.selectbox("Select Group By Column", ['None'] + filtered_cat_var)

        # Numerical variable plots
        if col in numeric_var:
            # Layout
            fig = make_subplots(
                rows = 2, cols = 2,
                subplot_titles = ("Histogram", "Boxplot", "Violin Plot", "Strip Plot")
            )

            # Density Plot (instead of histogram for grouped data)
            if zvar == 'None':
                fig.add_trace(
                    go.Histogram(
                        x=data[col],
                        name="Histogram",
                        marker=dict(color="#C0392B"),
                        showlegend=True  # Enable legend for this trace
                    ),
                    row = 1, col = 1
                )
            else:
               for group in data[zvar].unique():
                    print(f'line: 149 {group}') ##############################
                    # Create a density plot for each group
                    group_data = data.loc[data[zvar] == group][col]
                    print('line: 152')
                    fig.add_trace(
                        go.Histogram(
                        x = group_data,
                        histnorm='density',  # Normalize to show density
                        name=group,  # Legend name
                        opacity=0.6,
                        nbinsx = 50  # Set transparency
                    ),
                row=1, col=1
                    )

            fig.update_xaxes(title_text=col, row=1, col=1)
            fig.update_yaxes(title_text="Density", row=1, col=1)

            # Boxplot
            if zvar == 'None':
                fig.add_trace(
                    go.Box(
                        y=data[col],
                        name="Boxplot",
                        boxmean='sd',
                        showlegend=True
                    ),
                    row=1, col=2
                )
            else:
                for group in data[zvar].unique():
                    fig.add_trace(
                        go.Box(
                            y=data[data[zvar] == group][col],
                            name=f"{group} (Boxplot)",
                            boxmean='sd',
                            showlegend=True
                        ),
                        row=1, col=2
                    )

            fig.update_yaxes(title_text="Value", row=1, col=2)

            # Violin Plot
            if zvar == 'None':
                fig.add_trace(
                    go.Violin(
                        y=data[col],
                        box_visible=True,
                        line_color="orange",
                        name="Violin Plot",
                        showlegend=True
                    ),
                    row=2, col=1
                )
            else:
                for group in data[zvar].unique():
                    fig.add_trace(
                        go.Violin(
                            y=data[data[zvar] == group][col],
                            name=f"{group} (Violin)",
                            box_visible=True,
                            showlegend=True
                        ),
                        row=2, col=1
                    )

            fig.update_yaxes(title_text="Value", row=2, col=1)

            # Strip Plot
            if zvar == 'None':
                fig.add_trace(
                    go.Scatter(
                        x=data[col],
                        y=[1] * len(data),
                        mode="markers",
                        marker=dict(color='#82E0AA'),
                        name="Strip Plot",
                        showlegend=True
                    ),
                    row=2, col=2
                )
            else:
                for group in data[zvar].unique():
                    fig.add_trace(
                        go.Scatter(
                            x=data[data[zvar] == group][col],
                            y=[1] * len(data[data[zvar] == group]),
                            mode="markers",
                            name=f"{group} (Strip)",
                            marker=dict(
                                opacity=0.7
                            ),
                            showlegend=True
                        ),
                        row=2, col=2
                    )

            fig.update_yaxes(title_text="", row=2, col=2)

            # Update layout
            fig.update_layout(
                title={
                    'text': "Univariate Analysis of " + col,
                    'x': 0.5,
                    'y': 0.97,
                    'xanchor': 'center',
                    'yanchor': 'top',
                },
                height=800,
                width=1000
            )

            # Display the figure in Streamlit
            st.plotly_chart(fig)

        
        else:
            print('line: 295')
            # Categorical variable plots
            if zvar == 'None':
                # Bar Plot without grouping
                bar_fig = px.bar(
                    data,
                    y=data[col].value_counts().index,
                    x=data[col].value_counts().values,
                    labels={col: "Count", "y": col},
                    title="Bar Plot of " + col
                )
                st.plotly_chart(bar_fig)

                # Pie Chart without grouping
                pie_fig = px.pie(
                    data,
                    names=data[col].value_counts().index,
                    values=data[col].value_counts().values,
                    title="Pie Chart of " + col
                )
                st.plotly_chart(pie_fig)

            else:
                # Bar Plot with grouping by zvar
                grouped_data = data.groupby([zvar, col]).size().reset_index(name='Count')
                bar_fig = px.bar(
                    grouped_data,
                    x=zvar,
                    y="Count",
                    color=col,
                    barmode="group",
                    labels={"Count": "Frequency"},
                    title=f"Bar Plot of {col} Grouped by {zvar}"
                )
                st.plotly_chart(bar_fig)

                # Combine `zvar` and `col` into a single label for unique colors
                # Facet the pie chart by zvar
                pie_fig = px.pie(
                    grouped_data,
                    names=col,          # Slice by the selected column
                    values="Count",     # Use the count for proportions
                    facet_col=zvar,     # Create separate pies for each category in zvar
                    title=""
                )

                # Update layout for better display
                pie_fig.update_layout(
                    title={
                        'text': f"Pie Chart of {col} Grouped by {zvar}",
                        'x': 0.15,
                        'y': 0.95,
                        'xanchor': 'center',
                        'yanchor': 'top',
                    },
                    showlegend=True  # Keep legends visible for each pie
                )
                st.plotly_chart(pie_fig)




# Tab 3: Bivariate Plots with Subtabs for Numerical and Categorical
with tab3:
    st.write("### Bivariate Analysis")
    bivariate_tab1, bivariate_tab2 = st.tabs(["Numerical", "Categorical"])
    if data is not None:
        # Numerical Subtab
        with bivariate_tab1:
            x_col = st.selectbox("X-Axis", numeric_var)
            filtered_var = [c for c in numeric_var if c != x_col]
            y_col = st.selectbox("Y-Axis", filtered_var)
            filtered_zvar = [c for c in data.columns if c != y_col and c!= x_col]
            group_col = st.selectbox("Group By (Optional)", ["None"] + filtered_zvar)

            if group_col == "None":
                group_col = None

            # Numerical Bivariate Plot (Regression Line and Contour Plot)
            if data[x_col].dtype in [np.float64, np.int64] and data[y_col].dtype in [np.float64, np.int64]:
                # Scatter plot with Regression Line
                fig = px.scatter(data, x=x_col, y=y_col, color=group_col, trendline="ols", title=f"Scatter plot with regression line between {y_col} and {x_col}"
                                + (f" (grouped by {group_col})" if group_col else ""))
                st.plotly_chart(fig)

            # Create a color palette for groups
            group_colors = px.colors.qualitative.Set1
            
            # Create the Seaborn plot
            fig, ax = plt.subplots(figsize=(6, 3))
            
            # Check if group_col is selected
            if group_col:

                if group_col in cat_var:
                    unique_groups = data[group_col].unique()
                    palette = sns.color_palette("tab10", len(unique_groups))  # Use distinct colors for groups
                    
                    for i, group in enumerate(unique_groups):
                        group_data = data[data[group_col] == group]
                        sns.scatterplot(x=group_data[x_col], y = group_data[y_col], s = 5, color = palette[i], ax = ax, label=group)
                        sns.histplot(x=group_data[x_col], y = group_data[y_col], bins = 50, pthresh = 0.1, cmap = "coolwarm", 
                                    ax=ax, cbar = False)
                        sns.kdeplot(x=group_data[x_col], y = group_data[y_col], levels = 5, color = "w", linewidths = 1, ax = ax)

                    ax.set_title(f"Heatmap of {y_col} and {x_col} group by {group_col}")
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
                    st.pyplot(fig)

                else:
                    st.warning('Select a categorical variable as Group by for the Contour plot')
                

            else:

            # Contour plot
                contour_fig = go.Figure()

                if group_col is None:
                    # For ungrouped data, use a single colorscale like 'Viridis'
                    group_data = data[[x_col, y_col]].dropna()
                    xy_data = np.vstack([group_data[x_col], group_data[y_col]])
                    xx, yy = np.meshgrid(
                        np.linspace(group_data[x_col].min(), group_data[x_col].max(), 100),
                        np.linspace(group_data[y_col].min(), group_data[y_col].max(), 100)
                    )
                    zz = stats.gaussian_kde(xy_data)(np.vstack([xx.ravel(), yy.ravel()]))
                    zz = zz.reshape(xx.shape)

                    contour_fig.add_trace(go.Contour(
                        z=zz,
                        x=xx[0],
                        y=yy[:, 0],
                        colorscale="Viridis",  # Use a predefined colorscale for ungrouped data
                        contours_coloring="heatmap",
                        name="Density"
                    ))
                
                    

                    # Update the layout for the contour plot
                    contour_fig.update_layout(
                        title=f"Contour plot between {y_col} and {x_col}" + (f" (grouped by {group_col})" if group_col else ""),
                        xaxis_title=x_col,
                        yaxis_title=y_col,
                        showlegend=True,
                        legend_title=group_col if group_col else "Legend",
                        legend=dict(
                            orientation = "h",  # Horizontal legend
                            yanchor = "bottom",
                            y = -0.3,  # Adjusting the position of the legend
                            xanchor = "center",
                            x = 0.5
                        )
                    )

                    # Show the plot in Streamlit
                    st.plotly_chart(contour_fig)
                    

         # Categorical Bivariate Plot (Bar plot and Heatmap)
        with bivariate_tab2:
            x_col_cat = st.selectbox("X-Axis", cat_var)
            filtered_var = [c for c in cat_var if c != x_col_cat]
            y_col_cat = st.selectbox("Y-Axis", filtered_var)
            filtered_zvar = [c for c in filtered_var if c != y_col_cat]
            group_col_cat = st.selectbox("Group By (Optional)", ["None"] + filtered_zvar)

            if group_col_cat == "None":
                group_col_cat = None

            # Categorical Bivariate Plot (Heatmap)
            if group_col_cat is not None:
                # Get unique groups
                unique_groups = data[group_col_cat].unique()

                # Create subplots with a number of rows equal to the number of unique groups
                rows = len(unique_groups)
                fig_heatmap = make_subplots(
                    rows=rows, cols=1,
                    shared_xaxes=True, shared_yaxes=True,
                    vertical_spacing=0.2,  # Add some space between subplots
                    subplot_titles=[f"Group: {group}" for group in unique_groups]  # Set titles for each subplot
                )

                # Generate heatmap for each group
                for idx, group in enumerate(unique_groups):
                    group_data = data[data[group_col_cat] == group]
                    heatmap_data = pd.crosstab(group_data[x_col_cat], group_data[y_col_cat])

                    # Create the heatmap for the current group
                    heatmap = go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='Viridis',
                        colorbar=dict(
                    title="Count",
                    len=0.6,  # Adjust this for the color bar's length (height)
                    yanchor="top",
                    tickvals=[0, 1]
                   # ticktext=['Low', 'Medium', 'High']
                ),                
                        showscale=True,
                        name=f"Group: {group}"
                    )

                    # Add the heatmap to the corresponding subplot (idx + 1)
                    fig_heatmap.add_trace(heatmap, row=idx + 1, col=1)

                # Layout adjustments for heatmap
                fig_heatmap.update_layout(
                    title=f"Heatmap of {x_col_cat} vs {y_col_cat} grouped by {group_col_cat}",
                    height=300 * rows,  # Adjust height based on the number of rows (groups)
                    coloraxis_colorbar=dict(
                                            title="Count",
                                            len=0.3,  # Length of colorbar
                                            yanchor="top",  # Position colorbar to the top
                                            tickvals=[0, 0.5, 1],  # Example of placing ticks at 0, 0.5, and 1
                                            ticktext=['Low', 'Medium', 'High']  # Label the ticks
                                        ),
                showlegend=True,  # Display a single legend
                legend=dict(
                orientation='h',  # Horizontal legend
                yanchor='bottom',
                y=-0.2,  # Place legend below the plot
                xanchor='center',
                x=0.5,
                traceorder='normal',  # Normal legend order
                font=dict(size=12)  # Legend font size
            ),
                coloraxis_showscale=True
                )

                # Show the plot
                st.plotly_chart(fig_heatmap)

            else:
                try:
                    # If no grouping, just a single heatmap
                    heatmap_data = pd.crosstab(data[y_col_cat], data[x_col_cat])
                    fig_heatmap, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="viridis", ax=ax)
                    st.pyplot(fig_heatmap)
                except Exception as e:
                    if (x_col_cat is None) or (y_col_cat is None):
                        st.warning('There are not enough categorical variables')
#                    st.error(f"Error creating heatmap: {e}")

with tab4:
    if data is not None:
        group_col_cat_multi = st.selectbox("Group By (Optional)", ["None"] + cat_var, key = 'multi')
        
        if group_col_cat_multi == 'None':
            multi_plot = sns.pairplot(data = data[numeric_var])
        
        else:
            data_multi = data[numeric_var]
            data_multi[group_col_cat_multi] = data[group_col_cat_multi]
            print(data_multi.columns)
            multi_plot = sns.pairplot(data=data_multi, hue = group_col_cat_multi)

        st.pyplot(multi_plot.figure)



