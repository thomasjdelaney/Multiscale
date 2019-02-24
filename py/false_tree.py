"""
For building the wrong hierarchical structure in a tree.
"""
import anytree as at

# building trees TODO: move this to a separate script and save down using YAML
country_0 = at.Node('country_0')
country_1 = at.Node('country_1')
province_0 = at.Node('province_0', parent=country_0)
province_1 = at.Node('province_1', parent=country_0)
province_2 = at.Node('province_2', parent=country_0)
province_3 = at.Node('province_3', parent=country_1)
province_4 = at.Node('province_4', parent=country_1)
province_5 = at.Node('province_5', parent=country_1)
region_0 = at.Node('region_0', parent=province_0)
region_1 = at.Node('region_1', parent=province_0)
region_2 = at.Node('region_2', parent=province_0)
region_3 = at.Node('region_3', parent=province_0) # HERE IS THE CHANGE
region_4 = at.Node('region_4', parent=province_1)
region_5 = at.Node('region_5', parent=province_1)
region_6 = at.Node('region_6', parent=province_1)
region_7 = at.Node('region_7', parent=province_2)
region_8 = at.Node('region_8', parent=province_2)
region_9 = at.Node('region_9', parent=province_2)
region_10 = at.Node('region_10', parent=province_2)
region_11 = at.Node('region_11', parent=province_2)
region_12 = at.Node('region_12', parent=province_3)
region_13 = at.Node('region_13', parent=province_3)
region_14 = at.Node('region_14', parent=province_3)
region_15 = at.Node('region_15', parent=province_3)
region_16 = at.Node('region_16', parent=province_3)
region_17 = at.Node('region_17', parent=province_4)
region_18 = at.Node('region_18', parent=province_4)
region_19 = at.Node('region_19', parent=province_4)
region_20 = at.Node('region_20', parent=province_4)
region_21 = at.Node('region_21', parent=province_5)
region_22 = at.Node('region_22', parent=province_5)
region_23 = at.Node('region_23', parent=province_5)
