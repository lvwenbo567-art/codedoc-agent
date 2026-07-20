-- 1. 查询所有项目
SELECT *
FROM projects;

-- 2. 只查询项目核心字段
SELECT
    id,
    project_path,
    file_count,
    chunk_count,
    created_at
FROM projects;

-- 3. 查询指定项目
SELECT *
FROM projects
WHERE id = ?;

-- 4. 查询指定项目的文件
SELECT *
FROM files
WHERE project_id = ?;

-- 5. 查询所有 Python 文件
SELECT *
FROM files
WHERE suffix = '.py';

-- 6. 查询指定项目中的 Python 文件
SELECT *
FROM files
WHERE project_id = ?
  AND suffix = '.py';

-- 7. 查询指定项目的 chunks
SELECT *
FROM chunks
WHERE project_id = ?;

-- 8. 查询代码类型 chunks
SELECT *
FROM chunks
WHERE chunk_type = 'code';

-- 9. 查询长度大于等于指定值的 chunks
SELECT *
FROM chunks
WHERE length >= ?;

-- 10. 按长度从大到小排序
SELECT *
FROM chunks
ORDER BY length DESC;

-- 11. 查询长度最大的前 5 个 chunks
SELECT *
FROM chunks
ORDER BY length DESC
LIMIT 5;

-- 12. 分页查询 chunks
SELECT *
FROM chunks
ORDER BY id ASC
LIMIT ? OFFSET ?;

-- 13. 插入项目
INSERT INTO projects (
    project_path,
    file_count,
    chunk_count
)
VALUES (?, ?, ?);

-- 14. 修改项目统计
UPDATE projects
SET
    file_count = ?,
    chunk_count = ?
WHERE id = ?;

-- 15. 删除指定 chunk
DELETE FROM chunks
WHERE id = ?;
