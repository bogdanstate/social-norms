library('data.table')
library('rjson')

data <- fread('dyad_dataset.txt', header=F)
setnames(data, c('dyad', 'acts'))
data[, dyad:=gsub('(', '[', dyad, fixed=T)]
data[, dyad:=gsub(')', ']', dyad, fixed=T)]
data[, dyad:=gsub("'", '"', dyad, fixed=T)]
data[, dyad:=lapply(dyad, fromJSON)]
data[, user1:=sapply(dyad, function(x) x[1])]
data[, user2:=sapply(dyad, function(x) x[2])]

dt <- fread('graph_emb_shrunk.txt')
dt <- subset(dt, user != '')
head(dt)

dt.sub <- dt[order(pred_score * -1),.(user, pred_score, num_days)]
# dt.sub[,pred_score:=log(num_days+1)]
dt1 <- dt.sub[,.(user1=user, pred_score1=pred_score, num_days1=num_days)]
dt2 <- dt.sub[,.(user2=user, pred_score2=pred_score, num_days2=num_days)]
setkey(dt1, user1)
setkey(dt2, user2)

setkey(data, user1)
data <- merge(data, dt1)
setkey(data, user2)
data <- merge(data, dt2)

data[, dyad:=NULL]
write.table(data, file='dyad_dataset_for_lstm.txt', quote=F, row.names=F, col.names=F, sep='\t')
