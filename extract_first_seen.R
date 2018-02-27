library('data.table')
data <- fread('graph.txt')
setnames(data, names(data), c('id1', 'id2', 'num_interactions', 'start_time', 'end_time', 'num_distinct_days'))
data <- rbind(data[,.(ego=id1, start_time)], data[,.(ego=id2, start_time)])
data <- data[,.(start_time=min(start_time)),by=ego]
write.table(data, file='start_times.txt', sep='\t', col.names=F, row.names=F)
