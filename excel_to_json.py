import xlrd, re, json, os, codecs

sourcePath = 'source'
jsonPath = 'json'

reg = re.compile( '^(\w*)$' )
regArr1 = re.compile( r'''
    ^(\w*)
    \[(\w*)\]$
    ''', re.VERBOSE )
regArr2 = re.compile( r'''
    ^(\w*)
    \[(\d*)\]
    <(\w*)>$''', re.VERBOSE )
nType = 0

def XlsToJson( filePath, savePath ):
    print( 'filePath: ' + filePath + ' savePath: ' + savePath )
    data = xlrd.open_workbook( filePath )
    table = data.sheet_by_index( 0 )
    commentInfo = []
    headerInfo = []
    xlsInfo = {}
    dataInfo = {}
    ids = []

    dataInfo["dic"] = xlsInfo
    for i in range( table.ncols ):
        commentInfo.append( table.cell( 0, i ).value )
        header = table.cell( 1, i ).value
        headerInfo.append( header )
        match = regArr2.match( header )
        if match:
            nType = 3
        else:
            match = regArr1.match( header )
            if match:
                nType = 2
            else:
                match = reg.match( header )
                nType = 1
        nId = 0
        for j in range( 2, table.nrows ):
            cellInfo = table.cell( j, i ).value
            if cellInfo != "":
                if i == 0:
                    cellInfo = int( cellInfo )
                    ids.append( cellInfo )
                    xlsInfo[cellInfo] = {}
                nId = int(ids[j - 2])
                if nType == 1:
                    xlsInfo[nId][match.group( 1 )] = cellInfo
                elif nType == 2:
                    if match.group( 1 ) not in xlsInfo[nId]:
                        xlsInfo[nId][match.group( 1 )] = {}
                    xlsInfo[nId][match.group( 1 )][int( match.group( 2 ) )] = cellInfo
                elif nType == 3:
                    if match.group( 1 ) not in xlsInfo[nId]:
                        xlsInfo[nId][match.group( 1 )] = []
                    if len( xlsInfo[nId][match.group( 1 )] ) < int( match.group( 2 ) ):
                        xlsInfo[nId][match.group( 1 )].append( {} )
                    xlsInfo[nId][match.group( 1 )][int( match.group( 2 ) ) - 1][match.group( 3 )] = cellInfo

    outFile = codecs.open( savePath, 'w', 'utf-8' )

    json.dump( dataInfo, outFile, ensure_ascii=False, indent = 4, sort_keys = True)
    outFile.write('\n')

for root, dirs, files in os.walk( sourcePath ):
    for file in files:
        if file.endswith('.xls'):
            XlsToJson( os.path.join( root, file ), os.path.join( jsonPath, os.path.basename( file ).replace( '.xls', '.json' ) ) )