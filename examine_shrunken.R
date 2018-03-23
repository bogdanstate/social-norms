library('data.table')
library('igraph')
library('ggnetwork')
library('ggplot2')

dt <- fread('graph_emb_shrunk.txt')
dt <- subset(dt, user != '')
NUM_USERS <- 10000
dt.sub <- dt[order(pred_score * -1),.(user, pred_score, num_days)]
# dt.sub[,pred_score:=log(num_days+1)]
dt1 <- dt.sub[,.(user1=user, pred_score1=pred_score, num_days1=num_days)]
dt2 <- dt.sub[,.(user2=user, pred_score2=pred_score, num_days2=num_days)]
setkey(dt1, user1)
setkey(dt2, user2)

data <- fread('../ubuntu-corpus/graph.txt')
setnames(data, names(data), c('user1', 'user2', 'num_interactions', 'min_timestamp', 'max_timestamp', 'num_days'))
setkey(data, user1)
data <- merge(data, dt1)
setkey(data, user2)
data <- merge(data, dt2)
data[, pred_score_avg:=apply(cbind(pred_score1, pred_score2), c(1), min)]

data <- subset(data, num_days >= 3)

g <- graph.data.frame(data, directed=F, vertices=dt.sub)

com <- igraph::components(g)
max.com <- which(com$csize == max(com$csize))
g <- igraph::delete.vertices(g, V(g)[com$membership != max.com])
n <- ggnetwork(g, arrow.gap=0)
p <- ggplot(n, aes(x=x, y=y, xend=xend, yend=yend))
p <- p + geom_edges(aes(color=pred_score_avg), curvature=0.1, alpha=0.5)
p <- p + geom_nodes(aes(color=pred_score), size=1, alpha=0.5)
p <- p + ggtitle('')
p <- p + theme_blank()
p <- p + theme(text=element_text(size=22))
p <- p + theme(legend.position="bottom")
p <- p + scale_color_gradient2(low='blue', mid='gray', high='red', midpoint=median(V(g)$pred_score))
p <- p + guides(color=guide_colorbar(title='Ln Predicted days on site', title.position="top",
                                     barwidth=40))

png('network_shrunken.png', width=1000, height=1000)
print(p)
dev.off()
