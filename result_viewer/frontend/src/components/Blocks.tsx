// 安全内容块渲染(文档三 §5.1):只渲染后端提取的受控块类型;
// 不使用 dangerouslySetInnerHTML,不执行任何 HTML/MDX。

import type { Block } from '../app/types'

export function Blocks({ blocks }: { blocks: Block[] }) {
  return (
    <div className="doc-blocks">
      {blocks.map((b, i) => (
        <BlockView key={i} block={b} />
      ))}
    </div>
  )
}

function BlockView({ block }: { block: Block }) {
  switch (block.type) {
    case 'paragraph':
      return <p>{inlineText(block)}</p>
    case 'subheading':
      return <h4>{block.text}</h4>
    case 'list':
      return block.ordered ? (
        <ol>
          {(block.items ?? []).map((it, i) => (
            <li key={i}>{it}</li>
          ))}
        </ol>
      ) : (
        <ul>
          {(block.items ?? []).map((it, i) => (
            <li key={i}>{it}</li>
          ))}
        </ul>
      )
    case 'code':
      return (
        <pre>
          <code>{block.text}</code>
        </pre>
      )
    case 'table':
      return (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {(block.header ?? []).map((h, i) => (
                  <th key={i}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(block.rows ?? []).map((row, ri) => (
                <tr key={ri}>
                  {row.map((cell, ci) => (
                    <td key={ci}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    default:
      return null
  }
}

function inlineText(block: Block): React.ReactNode {
  const text = block.text ?? ''
  const links = block.links ?? []
  if (links.length === 0) return text
  // 段落内链接:仅当链接文字完整出现在段落中时渲染为 <a>
  const nodes: React.ReactNode[] = []
  let rest = text
  let key = 0
  for (const link of links) {
    const href = link.href
    if (!href || href.startsWith('#')) continue
    const m = /(.*?)\[(.+?)\]\([^)]*\)(.*)/s.exec(rest)
    if (!m) continue
    nodes.push(m[1])
    nodes.push(
      <a key={key++} href={href} target="_blank" rel="noreferrer noopener">
        {m[2]}
      </a>,
    )
    rest = m[3]
  }
  nodes.push(rest)
  return nodes
}
