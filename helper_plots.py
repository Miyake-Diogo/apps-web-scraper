## Functions to plot some charts in app
def plot_5_notes(dataframe):
    notas_5 = dataframe.query("score==5")["year_month"]
    fig, ax = plt.subplots()
    ax.hist(notas_5, bins=20, color='#33FFE9')
    plt.xticks(rotation='vertical')
    return st.pyplot(fig)