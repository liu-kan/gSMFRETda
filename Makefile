# chaneg docOutDir to what u want.
docOutDir = /Users/lp1/data/build/gSMFRETda
doc:
	( cat Doxyfile ; echo "OUTPUT_DIRECTORY=$(docOutDir)" ) | doxygen -

.PHONY: doc
