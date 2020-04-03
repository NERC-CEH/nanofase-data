#!/usr/bin/env bash
set -e

for yr in 2000 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020
do
	echo $yr
	./nanofase_data.py config_ag_$((yr)).yaml
	cp data.nc c:/users/sharrison/code/nanofase/data/thames/flat_ag_$((yr)).nc
done