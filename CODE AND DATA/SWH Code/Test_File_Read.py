from pathlib import Path
import sys

filepath = Path("createSyntheticData.py")

if not filepath.exists():
	raise Exception("Tried to read a file but the file does not exist.")

f = open(filepath.resolve(), "r", encoding="utf-8")

entireFileContents = ""
saveContentsTemp = ""
foundBeginMarker = False
saveLine = False
for line in f:
	entireFileContents += line
	if line.strip() == "#### BEGIN BACKUP REGION ####":
		saveLine = True
		foundBeginMarker = True
		continue
	if foundBeginMarker:
		if line.strip() == "#### END BACKUP REGION ####":
			break
	if saveLine:
		saveContentsTemp += line

f.close()

if foundBeginMarker:
	saveContents = saveContentsTemp
else:
	saveContents = entireFileContents


# print(saveContents)
print(sys.argv[0])

fWrite = open("aaaasdf.py.bak", "w", encoding="utf-8")
fWrite.write(saveContents)
fWrite.close()