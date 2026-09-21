// 把构建标识写入 frontend_dist/build_id.json,后端 /api/v1/version 会读取(文档三 §5.3 ui_build_id)
import { writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const dist = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'frontend_dist')
const buildId = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14)
writeFileSync(join(dist, 'build_id.json'), JSON.stringify({ build_id: buildId }, null, 2) + '\n')
console.log('build_id:', buildId)
