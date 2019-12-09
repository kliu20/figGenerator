import sys
import pandas as pd
import os
import getopt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from os import walk

#Path for output file
filePath = ''
#Column needs to be sorted/plot
col = ''
#Which type of figure to be generated
figType = ''
#Input directory passing from command argument
inputDir = ''
#Input mean path passing from command argument
meanPath = ''
#Fixed group number
numGroup = 5.0
#Predefined frequencies
frequencies = [1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2]
#Data range for Y axes
yRange = []
#Map of number of entries for each frequency of a group
groupCountMap = []
#Total number of elements
allNum = 0
#Threashold for each frequency
threshNum = []
#Store attributes
attr = []
#Record the number of rows that are dropped after shrinking
dropped = np.zeros(len(frequencies))
#Total value for each frequency
totalFreq = []
i = 0
#Pick top N methods in terms of aggregated energy consumption
TOPN = 5

dataFrame = pd.DataFrame()

class FigAttr:
	def __init__(self, minVal, maxVal, interval):
		self.minVal = minVal
		self.maxVal = maxVal
		self.interval = interval

#Current python working directory path
def path():
	cwd = os.getcwd()
	return cwd + '/'

#Add file names
def getDataFrame(path):
	files = []
	global dataFrame
	dataFrame = pd.read_csv(path)
	return dataFrame 

#Get column names 
def getCol(fileName):
	with open(fileName) as f:
		fileLine = f.readline()	
	return fileLine.split(",")

def readFirstMethodName(fileName):
	with open(fileName) as f:
		fileLine = f.readline()	#Read column name
		fileLine = f.readline() #Read the first row of contents
		return fileLine.split(",")[0]

#Workaround for sunflow
def readFirstMethodNameRawData(fileName):
#	with open(fileName) as f:
#		fileLine = f.readline()	#Read column name
#		fileLine = f.readline() #Read the first row of contents
#		return fileLine.split(",")[4]
	return 'org.sunflow.core.renderer.BucketRenderer.renderBucket'

"""
Get top n methods in terms of aggregated energy consumption for each frequency
Return a series of top n methodIDs
"""
def pickTopMethods(df, n):
	topN = []
	for freq in frequencies:
		dataFrame = df[(df['Frequency'] == freq)]

		value = dataFrame.groupby(['MethodID', 'Frequency'])['Package'].sum()
		
		methodDf = value.nlargest(n).reset_index()
		topN.append(methodDf[(methodDf['Frequency'] == freq)]['MethodID'])

	return topN
	

#Process the commands
def commandProcess(argv):
	global inputDir
	global figType
	global col
	global meanPath
	try:
		opts, args = getopt.getopt(argv, "hi:t:c:", ["inputDir=", "figureType", "column="])
	except getopt.GetoptError:
		print('genFig.py -t <Figure Type> -i <Input directory> -t <heatmap/linegraph>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
			print('test.py -c <centroidNum> -i <inputDir> -t <heatmap/linegraph>')
			sys.exit()
		elif opt in ("-i", "--inputDir"):
			inputDir = arg
		elif opt in ("-t", "--figureType"):
			figType = arg
		elif opt in ("-c", "--column"):
			col = arg
			

def createYLegend(minVal, maxVal, interval): 
	global numGroup
	global yRange
	yRange = []
	localMax = minVal
	i = 0
	groupID = 0
	#print('min is {}, max is {}'.format(minVal, maxVal))
	while True:
		rangeBegin = localMax
		localMax += interval
		preIndex = i
		if rangeBegin >= 1:
			#Floating point precision problem
			if groupID == numGroup - 1:
				yRange.append(str(round(rangeBegin, 1)) + '-' + str(round(maxVal, 1)))
				break
			else:
				yRange.append(str(round(rangeBegin, 1)) + '-' + str(round(localMax, 1)))

		elif rangeBegin < 1:
			if groupID == numGroup - 1:
				yRange.append(str(round(rangeBegin, 3)) + '-' + str(round(maxVal, 3)))
				break
			else:
				yRange.append(str(round(rangeBegin, 3)) + '-' + str(round(localMax, 3)))
		groupID += 1

	yRange = list(reversed(yRange))

def getArray(dataFrame):
	global frequencies
	for groupID in range(int(numGroup)):
		groupedDF = dataFrame[(dataFrame['groupID'] == groupID)]
		freqCount = []
		for freq in frequencies:
			freqCount.append(groupedDF[(groupedDF['Frequency'] == freq)].shape[0])

		groupCountMap.append(freqCount)	
	return np.array(groupCountMap)

def getFreqDF(dataFrame):
	global frequencies
	#for freq in frequencies:
	#df = pd.DataFrame(data = dataFrame[(dataFrame['Frequency'] == 1.2)])
	#print(dataFrame.loc[dataFrame['Frequency'] == 1.2])
	#print(dataFrame.loc[dataFrame['hotMethodMin'] == 150])
	print(dataFrame)

#Set group ID for each entry 
def setGroupID(dataFrame, minVal, maxVal, interval):
	global frequencies
	global dropped
	freqID = 0;
	i = 0
	data = []
	indexes = []
	if 'groupID' in dataFrame.columns:
		del dataFrame['groupID']
	for freq in frequencies:
		localMax = minVal
		groupID = 0
		localMaxIndexFreq = max(dataFrame[(dataFrame['Frequency'] == freq)].index.tolist())
		print(localMaxIndexFreq)
		while localMax < maxVal and i <= localMaxIndexFreq:
			groupCount = 0
			localMax += interval
			preIndex = i
			
			if groupID == numGroup - 1:
				while i <= localMaxIndexFreq and dataFrame[col].iloc[i] <= maxVal:
					#This is final group, jump out of outer iteration after this loop
#					print('data frame index {} group {} frequency {} package value is: {} localMax is {}'.format(i, groupID, freq, dataFrame[col].iloc[i], localMax ))
					i += 1
				#Drop the left entries that are not included within maxVal value
				localMax = maxVal
			else:
				while i <= localMaxIndexFreq and dataFrame[col].iloc[i] <= localMax:
					#Iterate until find out the last index of {code: freq} in post-shrinked (if it's applicable). Otherwise it's last index of 
					# {code: freq} in original table
#					print('data frame index {} group {} frequency {} package value is: {} localMax is {}'.format(i, groupID, freq, dataFrame[col].iloc[i], localMax ))
					i += 1

			for j in range(preIndex, i):
				data.append(groupID)
				indexes.append(j)
			
			groupID += 1
		if i - 1 != localMaxIndexFreq:
			dropped[freqID] = localMaxIndexFreq - (i - 1)
			#print('end loop, i is {} end index of its frequency {} is {}'.format(i - 1, freq, localMaxIndexFreq))
		#The size of current frequency has been shrinked. Add dummy cells.
		for k in range(i, localMaxIndexFreq + 1):
			data.append(sys.maxsize)
			indexes.append(k)
		#Index of next frequency 
		i = localMaxIndexFreq + 1
		freqID += 1

	dataFrame['groupID'] = pd.Series(data, indexes)


#Discard small number of cells in heatmap
def getShrinkedDF(dataFrame, minVal, maxVal, interval, maxGID, minGID, generation):
	global numGroup
	global frequencies
	global threshNum
	global attr

	if maxGID == numGroup - 1 and minGID == 0:
	#if maxGID == numGroup - 1: 
		#print('maxVal after getColorCellByCol: {}, maxGID is: {}, minVal is: {}'.format(maxVal, maxGID, minVal))
		attr = [minVal, maxVal, interval]
		return 

	#Record group ID of grey cells for each frequency
	maxGreyGroupID = np.zeros(len(frequencies))
	minGreyGroupID = np.zeros(len(frequencies))
	for i in range(len(frequencies)):
		j = 0
		tempMax = 0 
		tempMin = sys.maxsize
		for gid in range(int(numGroup)):
			#Record group ID of grey cells (significant number of occurences)
			if len(dataFrame[(dataFrame['groupID'] == gid) & (dataFrame['Frequency'] == frequencies[i])]) > threshNum[i]:
				#print('frequency: {} group: {} value: {} is larger than threshold {}'.format(frequencies[i], gid, len(dataFrame[(dataFrame['groupID'] == gid)]), threshNum[i]))
				tempMax = max(tempMax, gid)
				tempMin = min(tempMin, gid)
				#print('tempMin is {}, frequency is: {} value: {} is larger than threshold {}'.format(tempMin, frequencies[i], len(dataFrame[(dataFrame['groupID'] == gid) & (dataFrame['Frequency'] == frequencies[i])]), threshNum[i]))

		maxGreyGroupID[i] = tempMax
		minGreyGroupID[i] = tempMin
	#Record maxVal group ID of grey cell as well as its frequency id 
	maxGID = max(maxGreyGroupID)
	minGID = min(minGreyGroupID)

	#print('maxGID is {} minGID is {}'.format(maxGID, minGID))
	lowerLimit = minVal + interval * minGID
	upperLimit = lowerLimit + interval * (maxGID + 1)
	newInterval = (upperLimit - lowerLimit) / numGroup

	#Set group IDs
	setGroupID(dataFrame, lowerLimit, upperLimit, newInterval)

#	heatMapDF = recreateDF(dataFrame)
#	createYLegend(minVal, maxVal, interval)
#
#	#Generate each generation of shrinking process
#	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'num', generation)
#	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'percent', generation)

	getShrinkedDF(dataFrame, lowerLimit, upperLimit, newInterval, maxGID, minGID, generation + 1)

def getDroppedDF(heatMapDF):
	droppedPercentDF = pd.DataFrame()
	droppedNumDF = pd.DataFrame()

	for i in range(len(frequencies)):
		#dropPercent="{:.0%}".format((float(totalFreq[i]) - float(heatMapDF[frequencies[i]].sum())) / float(totalFreq[i]))
		dropPercent=(float(totalFreq[i]) - float(heatMapDF[frequencies[i]].sum())) / float(totalFreq[i])
		dropNum =float(totalFreq[i]) - float(heatMapDF[frequencies[i]].sum()) 
		droppedPercentDF[frequencies[i]] = pd.Series(dropPercent, [0])
		droppedNumDF[frequencies[i]] = pd.Series(dropNum, [0])

		droppedPercentDF.rename(index={1: 'Dropped'})
		droppedNumDF.rename(index={1: 'Dropped'})
		
	return droppedPercentDF, droppedNumDF

def getMeanDF(heatMapDF):
	global frequencies
	print(heatMapDF)
	meanDF = pd.DataFrame()
	for i in range(len(frequencies)):
		mean = heatMapDF[frequencies[i]].sum() / heatMapDF.shape[0]
		meanDF[frequencies[i]] = pd.Series(mean,[0])
		meanDF.rename(index={1: 'Mean'})
		
	return meanDF


#Recreate data frame for heatmap generation
def recreateDF(dataFrame):
	global frequencies
	global numGroup
	global threshNum
	freqCounter = [[0 for i in range(int(numGroup))] for j in range(len(frequencies))]
	for i in range(len(frequencies)):
		for gid in range(int(numGroup)):
			freqCounter[i][gid] = dataFrame[(dataFrame['groupID'] == gid) & (dataFrame['Frequency'] == frequencies[i])].shape[0]
	heatMapDF = pd.DataFrame()
	for i in range(len(frequencies)):
		heatMapDF[frequencies[i]] = pd.Series(freqCounter[i], range(len(freqCounter[i])))

	print(heatMapDF)
	return heatMapDF

def plotFig(heatMapDF, droppedPercentDF, droppedNumDF, numOrPercent, generation):
	global filePath
	global frequencies
	global threshNum
	global totalFreq
	global numGroup
	global meanPath

	meanDF = getMeanDF(heatMapDF)
	print(meanDF)
	
	ax = []
	with sns.axes_style('white'):
		#Plus 2 for dropped row and average row
		gs = gridspec.GridSpec(int(numGroup) + 2, 10)
		ax.append(plt.subplot(gs[:-3, 2:-1]))
		if(numOrPercent == 'num'):
			#Dropped
			ax.append(plt.subplot(gs[-2, 2:-1]))
		
			#Mean
			ax.append(plt.subplot(gs[-1, 2:-1]))
		else:
			#Dropped
			ax.append(plt.subplot(gs[-1, 2:-1]))
		

	#Show absolute number 
		if numOrPercent == 'num':

			for i in range(len(frequencies)):
				trick = threshNum[i] + 10
				cmap, norm = mcolors.from_levels_and_colors([0, threshNum[i], trick], ['white', 'grey', 'grey'], extend='max')
				sns.heatmap(heatMapDF.mask(((heatMapDF == heatMapDF) | heatMapDF.isnull()) & (heatMapDF.columns != frequencies[i])), ax=ax[0], cbar=False, annot=True, fmt='g', cmap=cmap, norm=norm)

			#Dropped row
			for i in range(len(frequencies)):
				cmap, norm = mcolors.from_levels_and_colors([0, threshNum[i], trick], ['white', 'white', 'white'], extend='max')
				sns.heatmap(droppedNumDF.mask(((droppedNumDF == droppedNumDF) | droppedNumDF.isnull()) & (droppedNumDF.columns != frequencies[i])), ax=ax[1], cbar=False, annot=True, fmt='g', cmap=cmap, norm=norm)
			#Mean row
			for i in range(len(frequencies)):
				cmap, norm = mcolors.from_levels_and_colors([0, threshNum[i], trick], ['white', 'white', 'white'], extend='max')
				sns.heatmap(meanDF.mask(((meanDF == meanDF) | meanDF.isnull()) & (meanDF.columns != frequencies[i])), ax=ax[2], cbar=False, annot=True, fmt='g', cmap=cmap, norm=norm)

		#Show percentage 
		elif numOrPercent == 'percent':
			percentHeatMap = pd.DataFrame()
			for i in range(heatMapDF.shape[1]):
				percentHeatMap[frequencies[i]] = heatMapDF[frequencies[i]].values / float(totalFreq[i])

			for i in range(len(frequencies)):
				trick = threshNum[i] + 10
				cmap, norm = mcolors.from_levels_and_colors([0, threshNum[i] / float(totalFreq[i]), trick], ['white', 'grey', 'grey'], extend='max')
				sns.heatmap(percentHeatMap.mask(((percentHeatMap == percentHeatMap) | percentHeatMap.isnull()) & (percentHeatMap.columns != frequencies[i])), ax=ax[0], cbar=False, annot=True, fmt="2.1%", cmap=cmap, norm=norm)

			for i in range(len(frequencies)):
				cmap, norm = mcolors.from_levels_and_colors([0, threshNum[i], trick], ['white', 'white', 'white'], extend='max')
				sns.heatmap(droppedPercentDF.mask(((droppedPercentDF == droppedPercentDF) | droppedPercentDF.isnull()) & (droppedPercentDF.columns != frequencies[i])), ax=ax[1], cbar=False, annot=True, fmt="1.1%", cmap=cmap, norm=norm)
			
	# want a more natural, table-like display
	ax[0].xaxis.tick_top()

	ax[0].set_xticklabels(frequencies, minor=False)
	global yRange

	ax[0].set_yticklabels(yRange, minor=False, rotation=0)
	ax[1].set_yticklabels(['Dropped'], minor=False, rotation=0)
	ax[1].set_xticklabels([], minor=False)
	if(numOrPercent == 'num'):
		ax[2].set_yticklabels(['Mean'], minor=False, rotation=0)
		ax[2].set_xticklabels([], minor=False)

	fileName = filePath.split('.')[0]
	name = fileName.split('/')[-1]
	
	if generation == 0:
		ax[0].set_title('Prior Shrinking Range: Distribution of ' + col + ' while measuring ' + name, y=1.1)
	else:
		ax[0].set_title('After Shrinking Range: Distribution of ' + col + ' while measuring ' + name, y=1.1)

	cur_pos = []
#	for i in range(2):
#		cur_pos.append(ax[i].get_position())
#		ax[i].set_position([cur_pos[i].x0 + cur_pos[i].width * 0.15, cur_pos[i].y0 + cur_pos[i].height * 0.1, cur_pos[i].width * 0.75, cur_pos[i].height * 0.8])

	#plt.colorbar(heatmap)
	figName = path() + "data/sunflow/fig/" + col + '_' + name + '_HeatMap_DVFS_' + numOrPercent + '_' + str(generation)
	plt.savefig(figName + ".eps", format='eps')

# def getMeanDF(dataFrame):
# 	global frequencies
# 
# 	print(dataFrame.columns[0])
# 	meanVal = [[0 for i in range(len(frequencies))] for j in range(dataFrame.columns.size)]
# 	for i in range(6, dataFrame.columns.size):
# 		for j in range(len(frequencies)):
# 			temp = dataFrame[(dataFrame['Frequency'] == frequencies[j])]
# 			meanVal[i][j] = temp[dataFrame.columns[i]].sum() / temp[dataFrame.columns[i]].shape[0]
# 
# 	meanDF = pd.DataFrame()
# 	for i in range(dataFrame.columns.size):
# 		meanDF[dataFrame.columns[i]] = pd.Series(meanVal[i], range(len(meanVal[i])))
# 	return meanDF

def genHeatMap(dataFrame, col):
	global filePath
	global numGroup
	global frequencies
	global threshNum
	global totalFreq
	global attr
	#Formatting
	dataFrame['Frequency'] = dataFrame[['Frequency']].values / 1000000.0
	dataFrame = dataFrame.round({'Frequency': 1})
	topNMethodID = pickTopMethods(dataFrame, TOPN)
	#Choose the first method only
	#dataFrame = dataFrame[(dataFrame['MethodName'] == readFirstMethodNameRawData(filePath))]
	#print('length of freq col is {}, length of {} is {}'.format(len(dataFrame['Frequency']), col, len(dataFrame[col])))
	#Sort by given counter for each frequency
	dataFrame = dataFrame.sort_values(by = ['Frequency', col], ascending=[True, True])
	#Reset indexdataFrame[(dataFrame['Frequency'] == 1.2)]
	dataFrame = dataFrame.reset_index(drop=True)

	minVal = min(dataFrame[col])
	maxVal = max(dataFrame[col])
	interval = (maxVal - minVal) / numGroup

	#Set threshold count of entries for each frequency
	for freq in frequencies:
		#threshNum.append(len(dataFrame[(dataFrame['Frequency'] == freq)]) / numGroup)
		threshNum.append(len(dataFrame[(dataFrame['Frequency'] == freq)]) * 0.05)
	#print('Before getShrinkedDF maxVal is {}, minVal is {}'.format(maxVal, minVal))
# 	meanDF = getMeanDF(dataFrame)
	setGroupID(dataFrame, minVal, maxVal, interval)
	heatMapDF = recreateDF(dataFrame)
	for freq in frequencies:
		totalFreq.append(heatMapDF[freq].sum())

	droppedPercentDF, droppedNumDF = getDroppedDF(heatMapDF)

	createYLegend(minVal, maxVal, interval)

	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'num', 0)
	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'percent', 0)


	#Biggest groupd ID of grey cell
	maxGID = 0
	minGID = sys.maxsize
	generation = 1
	getShrinkedDF(dataFrame, minVal, maxVal, interval, maxGID, minGID, generation)
	#print('After  getShrinkedDF maxVal is {}, minVal is {}'.format(attr[1], attr[0]))

	heatMapDF = recreateDF(dataFrame)
	droppedPercentDF, droppedNumDF = getDroppedDF(heatMapDF)

	createYLegend(attr[0], attr[1], attr[2])

	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'num', 'last')
	plotFig(heatMapDF, droppedPercentDF, droppedNumDF, 'percent', 'last')
	#genSingleLineGraph(meanDF)


#def genSingleLineGraph(meanDF):
#	fig, ax = plt.subplots()
#	ax.set_title('Prior Shrinking Range: Distribution of ' + col + ' while measuring ' + name, y=1.1)
#	ax.set_ylabel(col, fontsize='medium')
#	cur_pos = ax.get_position()
#	ax.set_position([cur_pos.x0 + cur_pos.width * 0.1, cur_pos.y0 + cur_pos.height * 0.1, cur_pos.width * 0.75, cur_pos.height * 0.9])
#	meanDF.plot(x="freqCounter", y=getCol(col, legend=False, ax=ax, xticks=dataFrame['freqCounter'])
	#plt.legend([v[0] for v in meanDF['centroidNo']], loc='center left', bbox_to_anchor=(1.1, 0.5))
	#figName = path() + "data/sunflow/fig/" + col + '_' + name + '_LineGraph_DVFS_' + numOrPercent + '_last' 
	#plt.savefig(figName + ".eps", format='eps')

#Generate Line graph
def genLineGraph(dataFrame):
	global filePath
	#Formatting
	dataFrame['freqCounter'] = dataFrame[['freqCounter']].values / 1000000.0
	#Choose the first method only
	dataFrame = dataFrame[(dataFrame['methodName'] == readFirstMethodName(filePath))]
	
	fig, ax = plt.subplots(2, sharex=False);
	fileName = filePath.split("/")[-1]
	
	counterName = []
	counterName.append(getCol(filePath)[3].strip())
	counterName.append(getCol(filePath)[4].strip())
	
	name = fileName.split('_')
	if name[0] == 'all':
		fig.suptitle('XMeans Clustering for All Attributes', fontsize = 16)
	else:
		fig.suptitle('XMeans Clustering for ' + name[0] + ' Attribute', fontsize = 16)

	ax[0].set_title(name[1] + '_' + name[2] + '_' + 'DVFS')
	
	cur_pos = []
	for i in range(2):
		ax[i].set_ylabel(counterName[i], fontsize='medium')
		cur_pos.append(ax[i].get_position())
		ax[i].set_position([cur_pos[i].x0 + cur_pos[i].width * 0.1, cur_pos[i].y0 + cur_pos[i].height * 0.1, cur_pos[i].width * 0.75, cur_pos[i].height * 0.9])
		
	dataFrame.groupby("centroidNo").plot(x="freqCounter", y=getCol(filePath)[3].strip(), legend=False, ax=ax[0], xticks=dataFrame['freqCounter'])
	dataFrame.groupby("centroidNo").plot(x="freqCounter", y=getCol(filePath)[4].strip(), ax=ax[1], xticks=dataFrame['freqCounter'])

	plt.legend([v[0] for v in dataFrame.groupby('centroidNo')['centroidNo']], loc='center left', bbox_to_anchor=(1.1, 0.5))
	
	figName = path() + "data/sunflow/fig/" + fileName.split(".")[0]
	plt.savefig(figName + ".eps", format='eps')

	
def main(argv):	
	global filePath
	commandProcess(argv)
	filePath = path() + inputDir
	dataFrame = getDataFrame(filePath)
	if figType == 'line':
		genLineGraph(dataFrame)
	elif figType == 'heatmap':
		genHeatMap(dataFrame, col)


if __name__ == '__main__':
	main(sys.argv[1:])
