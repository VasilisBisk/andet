#!/bin/bash

network=$(docker network ls --format '{{ json .Name }}' | grep elastic)

if [ $network == "elastic" ]; then
    docker network create elastic
fi

docker run -d --name elasticsearch --net elastic -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:8.2.2
