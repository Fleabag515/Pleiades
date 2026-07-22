import { useState } from 'react'
import { avatarUrl } from '../lib/api'
import { initial } from '../lib/format'

interface AvatarProps {
  base: string
  character: string
  size?: number
}

/** Character avatar image with graceful fallback to an initial-letter badge
 * (GET /api/profiles/{name}/avatar 404s when no avatar has been set). */
function Avatar({ base, character, size = 32 }: AvatarProps): React.JSX.Element {
  const [failed, setFailed] = useState(false)
  const dim = { width: size, height: size, fontSize: Math.max(11, size * 0.42) }

  if (failed || !character) {
    return (
      <div
        style={dim}
        className="flex flex-none items-center justify-center rounded-full bg-accent/25 font-semibold text-accent"
      >
        {initial(character || '?')}
      </div>
    )
  }

  return (
    <img
      src={avatarUrl(base, character)}
      alt=""
      style={dim}
      className="flex-none rounded-full object-cover"
      onError={() => setFailed(true)}
    />
  )
}

export default Avatar
