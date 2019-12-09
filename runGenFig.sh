#!/bin/bash

genParams=('Package' 'Package_power' 'WallClockTime')
branchParams=('Branch-misses' 'Branches' 'BranchMissRate')
cacheParams=('Cache-misses' 'Cache-references' 'CacheMissRate')
cpuClockParams=('Cpu-clock' 'CPUClocksByTime')
cpuCycleParams=('Cpu-cycles' 'CPUCyclesByTime')
pageFaultParams=('Page-faults' 'PageFaultsByTime')



genLineGraph() {
	in_dir=data/sunflow/centroid/
	#shopt -s nullglob
	#inputFiles=($in_dir*.csv)
	inputFiles=$(ls $in_dir | grep "\.csv")
	#Generate line graph
	for file in ${inputFiles[@]}; do
		python genFig.py -t line -i $file
	done
}

genHeatMap() {
	in_dir=data/sunflow/rawData
	#shopt -s nullglob
	inputFiles=$(ls $in_dir | grep "\.csv")
	#Generate heat map
	for file in ${inputFiles[@]}; do
		IFS='_' read -ra NAME <<< "${file}"
		if [ ${NAME[0]} == 'branch-misses' ]; then
			for param in ${branchParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi

		if [ ${NAME[0]} == 'cache-misses' ]; then
			for param in ${cacheParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi

		if [ ${NAME[0]} == 'cpu-clocks' ]; then
			for param in ${cpuClockParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi

		if [ ${NAME[0]} == 'cpu-cycles' ]; then
			for param in ${cpuCycleParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi

		if [ ${NAME[0]} == 'page-faults' ]; then
			for param in ${pageFaultParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi
		if [ ${NAME[0]} == 'Energy' ]; then
			for param in ${genParams[@]}; do
				python genFig.py -t heatmap -c ${param} -i ${in_dir}/$file
			done
		fi
	done
	#python genFig.py -t heatmap -c Package_power -i $in_dir/branch-misses.csv
	#python genFig.py -t heatmap -c Package_power -i $in_dir/cache-misses.csv
}
	
#genLineGraph
genHeatMap


