-- 1. 每个项目的文件数量，使用 LEFT JOIN 保留没有文件的项目
SELECT
    p.id AS project_id,
    p.project_path,
    COUNT(f.id) AS file_count
FROM projects AS p
LEFT JOIN files AS f
    ON f.project_id = p.id
GROUP BY
    p.id,
    p.project_path
ORDER BY p.id ASC;

-- 2. 每个项目的 chunk 数量
SELECT
    p.id AS project_id,
    p.project_path,
    COUNT(c.id) AS chunk_count
FROM projects AS p
LEFT JOIN chunks AS c
    ON c.project_id = p.id
GROUP BY
    p.id,
    p.project_path
ORDER BY chunk_count DESC;

-- 3. 每种 chunk 类型的数量和长度统计
SELECT
    chunk_type,
    COUNT(*) AS chunk_count,
    ROUND(AVG(length), 2) AS average_length,
    MAX(length) AS maximum_length,
    MIN(length) AS minimum_length
FROM chunks
GROUP BY chunk_type
ORDER BY chunk_count DESC;

-- 4. 每个文件对应的 chunk 数量
SELECT
    f.id AS file_id,
    f.path,
    COUNT(c.id) AS chunk_count
FROM files AS f
LEFT JOIN chunks AS c
    ON c.project_id = f.project_id
   AND c.source_path = f.path
GROUP BY
    f.id,
    f.path
ORDER BY chunk_count DESC;

-- 5. 查询没有 chunk 的文件
SELECT
    f.id,
    f.path
FROM files AS f
LEFT JOIN chunks AS c
    ON c.project_id = f.project_id
   AND c.source_path = f.path
WHERE c.id IS NULL;

-- 6. 使用 HAVING 查询 chunk 数量大于指定值的项目
SELECT
    p.id AS project_id,
    p.project_path,
    COUNT(c.id) AS chunk_count
FROM projects AS p
INNER JOIN chunks AS c
    ON c.project_id = p.id
GROUP BY
    p.id,
    p.project_path
HAVING COUNT(c.id) > ?
ORDER BY chunk_count DESC;
