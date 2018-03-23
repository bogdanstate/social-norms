library('data.table')
library('igraph')
library('ggnetwork')
library('ggplot2')

data <- fread('/data/bogdan/ubuntu-corpus/graph.txt')
setnames(data, names(data), c('user1', 'user2', 'num_interactions', 'min_timestamp', 'max_timestamp', 'num_days'))

g <- graph.data.frame(data, directed=F)
pr <- page.rank(g) 
ec <- evcent(g) 
bc <- betweenness(g) 
