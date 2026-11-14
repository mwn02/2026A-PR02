from donnees import load_data, box_plot_bike_count
    
if __name__ == "__main__":
   df = load_data()
   box_plot_bike_count(df)