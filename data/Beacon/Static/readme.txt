This folder contains some data acquired with a smartphone from a set of 7 BEACONS.
In this readme, anchors refer to beacons, while the tag refers to the smartphone.

Each csv file corresponds to 1 set of measures which last during ~2min
	beacon_data_P0.csv
	beacon_data_P1.csv
	beacon_data_P2.csv
	beacon_data_P3.csv
	beacon_data_P4.csv

The anchors are never moving. Their positions remain the same for all csv file.
The tag is not moving during the measures, but its position differs from a csv file to another.

Positions and ids of the 7 anchors (contained in anchors.csv):
	ID	X	Y	Z
	236,	6.80,	6.87,	1.96
	237,	2.28,	9.60,	2.01
	238,	0.00,	6.63,	2.48
	239,	3.97,	0.00,	2.06
	240, 	3.67,	5.45,	2.83
	241,	0.00,	4.69,	2.22
	242,	6.80,	2.62,	2.92

The 5 positions of the tag for P0 to P4 (contained in real_position.csv):
	P0 = [3.11, 3.78, 1.78]
	P1 = [4.73, 4.16, 0.41]
	P2 = [5.66, 2.10, 0.75]
	P3 = [5.70, 7.38, 0.75]
	P4 = [0.68, 6.15, 0.75]

Each row of csv files corresponds to a single mesurement between the tag and 1 anchor.
The column of csv files refers to:
	<timestamp>, <tag_id>, <anchor_id>, NaN, NaN, NaN, <range>, <TxPower>, <RSSI>

	Where the timestamp is in milliseconds since epoch.
	The range is the distance in meters.
	TxPower and RSSI are in dbm.
	The range has been computed using the formula: range = 10^((TxPower-RSSI)/20)
	
For more information about the beacons and the scene we used, read the short report on the Ubiment switch drive:
    Reports/Beacon/Beacon.docx
