library('data.table')
library('igraph')
library('ggnetwork')
library('ggplot2')

data <- fread('graph.txt')
setnames(data, names(data), c('user1', 'user2', 'num_interactions', 'min_timestamp', 'max_timestamp', 'num_days'))
data <- subset(data, num_days >= 10)

careers <- fread('careers.txt')
setnames(careers, names(careers), c('user', 'day', 'i'))
careers <- careers[,.(num_days=nrow(.SD), first_day=min(day)),by=.(user)]
careers <- subset(careers, !grepl('bot', user))

setnames(careers, c('num_days', 'first_day'),  c('num_days1', 'first_day1'))
setkey(careers, user)
setnames(data, 'user1', 'user')
setkey(data, user)
data <- merge(data, careers)
setnames(data, c('user', 'user2'), c('user1', 'user'))
setkey(data, user)
setnames(careers, c('num_days1', 'first_day1'), c('num_days2', 'first_day2'))
setkey(careers, user)
data <- merge(data, careers)
setnames(data, 'user', 'user2')
setnames(careers, c('num_days2', 'first_day2'), c('num_days', 'first_day'))

data[, first_day_interaction:=as.Date(as.POSIXct(min_timestamp, origin='1970-01-01'))]
careers[, first_day:=as.Date(as.POSIXct(first_day, origin='1970-01-01'))]

g <- graph.data.frame(data, directed=F, vertices=careers)
com <- components(g)
max.com <- which(com$csize == max(com$csize))
g <- delete.vertices(g, V(g)[com$membership != max.com])

labels <- as.Date(c("2005-01-01", "2007-01-01", "2009-01-01", "2011-01-01", "2013-01-01"))
n <- ggnetwork(g)
p <- ggplot(n, aes(x=x, y=y, xend=xend, yend=yend))
p <- p + geom_edges(aes(color=first_day_interaction), curvature=0.1, alpha=0.3)
p <- p + geom_nodes(aes(color=first_day), size=3, alpha=0.5)
p <- p + scale_color_gradientn(colors=rainbow(5), breaks=as.integer(labels), labels=format(labels, "%Y"))
p <- p + ggtitle('')
p <- p + theme_blank()
p <- p + theme(text=element_text(size=22))
p <- p + theme(legend.position="bottom")
p <- p + guides(color=guide_colorbar(title='First day of posting (nodes) / interaction (edges)', title.position="top",
                                     barwidth=40))

png('network.png', width=1000, height=1000)
print(p)
dev.off()
