import MySQLdb

# configure db
hostname = "localhost"       
port = 3306
username = "root"
password = "111111"
dbname = "bdtune"           
tablename = "logInfo"  

def readDB():
  conn = MySQLdb.connect(host=hostname, port=port, user=username, passwd=password, db=dbname)
  cur = conn.cursor()
  sql = "select timestamp, syslogfacility, syslogfacilityText, syslogseverity, syslogseverityText, msg from " \
    + tablename + ";"
  cur.execute(sql)
  res = cur.fetchall()
  return res
  cur.close()
  conn.commit()
  conn.close()
  
def geneFile():
  with open(filename, 'w+') as wf:
    for line in readDB():
      line = '\t'.join([str(i).strip() for i in line])
      wf.write(line+'\n')
	  
if __name__ == '__main__':
  filename = 'log_dataset.txt'
  geneFile()
  print('log data set is generated !')