/** A Task-Manager/Mission-Center style scrolling live line graph: filled
 * area under a stroked line, newest sample on the right, fixed-height
 * gridlines. Pure SVG, no charting library (this app doesn't have one and
 * a rolling line graph doesn't need one) -- scales to `max` (100 for a
 * percentage, or an absolute value like RAM total for a byte-count graph).
 * `data` should already be capped to the desired rolling-window length by
 * the caller; this component just renders whatever it's given. */
function LiveGraph({
  data,
  max,
  height = 64,
  color = 'var(--color-accent)'
}: {
  data: number[]
  max: number
  height?: number
  color?: string
}): React.JSX.Element {
  const width = 200 // viewBox units; SVG scales to the container via CSS width=100%
  const safeMax = max > 0 ? max : 1
  const points = data.map((v, i) => {
    const x = data.length > 1 ? (i / (data.length - 1)) * width : width
    const y = height - (Math.min(v, safeMax) / safeMax) * height
    return [x, y] as const
  })
  const linePath = points.map(([x, y], i) => `${i === 0 ? 'M' : 'L'}${x.toFixed(2)},${y.toFixed(2)}`).join(' ')
  const areaPath = points.length
    ? `${linePath} L${width},${height} L0,${height} Z`
    : ''

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="w-full"
      style={{ height }}
    >
      {/* gridlines at 25/50/75% */}
      {[0.25, 0.5, 0.75].map((f) => (
        <line
          key={f}
          x1={0}
          x2={width}
          y1={height * f}
          y2={height * f}
          stroke="var(--color-bg-surface-hover)"
          strokeWidth={0.5}
        />
      ))}
      {areaPath && <path d={areaPath} fill={color} fillOpacity={0.15} stroke="none" />}
      {linePath && <path d={linePath} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />}
    </svg>
  )
}

export default LiveGraph
