#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:52:10 2020

@author: doorleyr
"""

# trip in
tn=this_world.tn
to_loc=this_world.geogrid.cells[100]
from_loc=this_world.zones[0]

test=tn.get_routes(from_loc, to_loc)

test['driving'].internal_route
test['driving'].internal_route['costs']

test['driving'].costs


# internal only
tn=this_world.tn
to_loc=this_world.geogrid.cells[100]
from_loc=this_world.geogrid.cells[20]
test=tn.get_routes(from_loc, to_loc)

test['driving'].internal_route
test['driving'].internal_route['costs']

test['driving'].costs


# trip out
tn=this_world.tn
from_loc=this_world.geogrid.cells[100]
to_loc=this_world.zones[200]

test=tn.get_routes(from_loc, to_loc)

test['driving'].internal_route
test['driving'].internal_route['costs']

test['driving'].costs