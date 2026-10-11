import { memo, useCallback, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CheckIcon, CopyIcon } from './Icons';

function CopyButton({ getText }: { getText: () => string }) {
  const [copied, setCopied] = useState(false);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(getText());
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard blocked — nothing useful to show */
    }
  }, [getText]);

  return (
    <button
      onClick={copy}
      className="absolute right-2 top-2 rounded-md border px-2 py-1 text-xs opacity-0
                 transition-opacity focus:opacity-100 group-hover:opacity-100"
      style={{
        backgroundColor: 'rgb(var(--bg-elevated))',
        borderColor: 'rgb(var(--border))',
        color: 'rgb(var(--text-muted))',
      }}
      aria-label={copied ? 'Copied' : 'Copy code'}
    >
      <span className="flex items-center gap-1">
        {copied ? <CheckIcon className="h-3 w-3" /> : <CopyIcon className="h-3 w-3" />}
        {copied ? 'Copied' : 'Copy'}
      </span>
    </button>
  );
}

/** A fenced block plus its copy affordance. */
function Pre({ children, ...rest }: React.HTMLAttributes<HTMLPreElement>) {
  const ref = useRef<HTMLPreElement>(null);
  const getText = useCallback(() => ref.current?.innerText ?? '', []);

  return (
    <div className="group relative">
      <pre ref={ref} {...rest}>
        {children}
      </pre>
      <CopyButton getText={getText} />
    </div>
  );
}

export const Markdown = memo(function Markdown({ content }: { content: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[[rehypeHighlight, { detect: true, ignoreMissing: true }]]}
        components={{
          pre: Pre as never,
          a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noreferrer noopener" />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
});
