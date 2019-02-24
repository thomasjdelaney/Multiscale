"""
For building dictionaries mapping countries or provinces to colours. 
"""
import matplotlib.cm as cm

province_to_colour = {}
colours = cm.gist_rainbow(np.linspace(0, 1, 6))
province_to_colour[province_0] = colours[0]
province_to_colour[province_1] = colours[1]
province_to_colour[province_2] = colours[2]
province_to_colour[province_3] = colours[3]
province_to_colour[province_4] = colours[4]
province_to_colour[province_5] = colours[5]

country_to_colour = {}
colours = cm.rainbow(np.linspace(0, 1, 2))
country_to_colour[country_0] = colours[0]
country_to_colour[country_1] = colours[1]
