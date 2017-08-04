features of each line in log_dataset:

timestamp, syslogfacility, syslogfacilityText, syslogseverity, syslogseverityText, msg

run:

1. log2dataset.py:      generate log txt file from mysql
2. dataset2label.py:    split log txt file into 128X time steps train_test text and label
3. 