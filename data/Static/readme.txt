This folder contains some data acquired with the pozyx system using 4 anchors and 1 tag.

Each csv file corresponds to 1 set of measures which last during ~1min
	pozyx_data_P0.csv
	pozyx_data_P1.csv
	pozyx_data_P2.csv
	pozyx_data_P3.csv
	pozyx_data_P4.csv

The anchors are never moving. Their positions remain the same for all csv file.
The tag is not moving during the measures, but its position differs from a csv file to another.

Positions of the 4 anchors:
	Anchor1 = [6.80, -2.74, 1.96]
	Anchor2 = [2.28,     0, 2.01]
	Anchor3 = [   0, -2.98, 2.48]    
	Anchor4 = [3.97, -9.60, 2.06]

IDs of the 4 anchors:
	anchor_ids = [28479, 28435, 28459, 28457]


The 5 positions of the tag:
	P0 = [3.11, -5.83, 1.78]
	P1 = [4.73, -5.44, 0.32]
	P2 = [5.66, -7.51, 1.07]
	P3 = [5.70, -2.23, 1.07]
	P4 = [0.68, -3.46, 1.07]

Each row of csv files corresponds to a single mesurement between the tag and 1 locator.
The column of csv files refers to:
	<timestamp>, <tag_id>, <locator_id>, NaN, NaN, NaN, <range>

	Where the timestamp is in milliseconds since epoch.
	And the range is the distance in meters.


For more information about the scene and setup, you can read the chapter "Multiple Range Measurements" of the Pozyx's report on the switchdrive:
	UbiMent\Reports\Pozyx\Pozyx.docx




