// JavaScript for MIMO Iterative Separation Website

document.addEventListener('DOMContentLoaded', () => {
  // Code Tab Switching
  const tabs = document.querySelectorAll('.code-tab');
  const codeBlocks = document.querySelectorAll('.code-block');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');

      tabs.forEach(t => t.classList.remove('active'));
      codeBlocks.forEach(b => b.classList.remove('active'));

      tab.classList.add('active');
      const targetBlock = document.getElementById(targetId);
      if (targetBlock) {
        targetBlock.classList.add('active');
      }
    });
  });

  // Copy Code Snippet
  const copyBtns = document.querySelectorAll('.copy-btn');
  copyBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const activeBlock = document.querySelector('.code-block.active pre');
      if (activeBlock) {
        const codeText = activeBlock.innerText;
        navigator.clipboard.writeText(codeText).then(() => {
          const originalText = btn.innerHTML;
          btn.innerHTML = '✓ Copied!';
          setTimeout(() => {
            btn.innerHTML = originalText;
          }, 2000);
        });
      }
    });
  });
});
