# Delete local branches that no longer exists on remote (windows command line)
git branch -vv | Where-Object { $_ -match 'gone\]' } | ForEach-Object { $_.Trim().Split()[0] } | ForEach-Object { git branch -D $_ }
