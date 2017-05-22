This folder contains some data acquired with the pozyx system using 4 anchors and 1 tag.

Each csv file corresponds to 1 set of measures which last during ~1min. For the same position, two set of data are acquired with different orientations (to get different april tag).
	pozyx_data_P0_1.csv
	pozyx_data_P0_2.csv
	pozyx_data_P1_1.csv
	pozyx_data_P1_2.csv
	pozyx_data_P2_1.csv
	pozyx_data_P2_2.csv
	pozyx_data_P3_1.csv
	pozyx_data_P3_2.csv
	pozyx_data_P4_1.csv
	pozyx_data_P4_2.csv

The anchors are never moving. Their positions remain the same for all csv file.
The tag is not moving during the measures, but its position differs from a csv file to another.

Positions of the 12 anchors:

id 		x		y 		z
28479	6.8		6.865	1.96
28435	2.28	9.605	2.01
28459	0		6.625	2.48
28457	3.97	0.005	2.06
	0	0		7.2242	2.5928
	1	0		4.8162	2.3428
	2	0		2.4072	2.6488
	3	3.2152	0		2.1918
	4	6.1492	0		1.6138
	5	6.802	3.2192	2.3428
	6	6.802	4.8962	2.2478
	7	7.046	7.7282	2.1898


The 5 positions of the tag:
x		y 		z
3.11	3.775	1.46
4.73	4.165	0
5.66	2.095	0.75
5.7		7.375	0.75
0.68	6.145	0.75

To descriminate data acquired from pozyx or from april tag we use respectively the tag_id 28486 and 333.

Each row of csv files corresponds to a single mesurement between the tag and 1 locator.
The column of csv files refers to:
	<timestamp>, <tag_id>, <locator_id>, NaN, NaN, NaN, <range>, <abs_error>, <var>

	Where the timestamp is in milliseconds since epoch.
	And the range is the distance in meters.
	The abs_error and variance are estimated in fonction of the range and orientation (for april tag) using regression made on previously acquired data.


For more information about the scene and setup, you can read the chapter "Multiple Range Measurements" of the Pozyx's report on the switchdrive:
	UbiMent\Reports\Pozyx\Pozyx.docx




