

def test(model, data, args):
    x_test, y_test = data
    y_pred, x_recon = model.predict(x_test, batch_size=10)
    print('-'*30 + 'Begin: test' + '-'*30)
    print('Test acc:', np.sum(np.argmax(y_pred, 1) == np.argmax(y_test, 1))/y_test.shape[0])
    img = combine_images(np.concatenate([x_test[:50],x_recon[:50]]))
    image = img * 255
    Image.fromarray(image.astype(np.uint8)).save(args.save_dir + "/real_and_recon"+str(args.iter)+".png")
    #Image.fromarray(image.astype(np.uint8)).filter(ImageFilter.SHARPEN).save(args.save_dir + "/real_and_recon"+str(args.iter)+".png")
    print('Reconstructed images are saved to %s/real_and_recon.png' % args.save_dir)
    print('-' * 30 + 'End: test' + '-' * 30)
    #plt.imshow(plt.imread(args.save_dir + "/real_and_recon.png"))
    #plt.show()